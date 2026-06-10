"""
kinetic_executor.py — Ontology/Vol-Monitor 시그널을 Alpaca 페이퍼 주문으로 실행하는 키네틱 레이어.

설계 (2026-06):
  트리거 : 온톨로지 추론 시그널 (ontology/trigger.py). 테일/크래시/변동성-리스크
           룰만 롱 스트래들로 매핑 (vol_overpriced·rate_* 제외). --source vol 로
           기존 vol_monitor_data.json(z>=2.5) 폴백 가능.
  액션   : 해당 ETF에 LONG STRADDLE (ATM 콜 매수 + ATM 풋 매수, ~30 DTE)
           — 방향을 예측하지 않고 변동성 자체를 먹는 명세서 핵심 thesis 구현.
  사이징 : 고정 비중(BASE_WEIGHT) × confidence강도 × 피드백승수 → 프리미엄 예산
           qty = floor(예산 / 스트래들 비용)
  저널   : 모든 진입/청산을 trade_journal.py 로 기록 (시그널 맥락 + 실현 P&L).
           귀인은 attribution.py, 피드백 승수는 feedback.py 가 생성.

모드:
  (없음)   dry-run 진입 미리보기 — 주문 미제출
  --live   진입 실제 발주 (장중에만)
  --exit   보유 스트래들 청산 점검 (익절/손절/만기임박)
  --auto   청산 → 진입 을 라이브로 연속 실행. 스케줄러용. 휴장 시 자동 no-op.

안전장치:
  - 기본 dry-run. 실제 발주는 --live / --auto 명시 필요.
  - 옵션 마켓 주문은 장중(09:30–16:00 ET)에만 가능 → 휴장 시 발주 중단.
  - 선물(=)·지수(^) 티커 자동 제외.
  - 중복 진입 방지: 이미 옵션 포지션 보유한 underlying 은 건너뜀.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
VOL_JSON = ROOT.parent / "macro-portal" / "public" / "data" / "vol_monitor_data.json"
DATA_BASE = "https://data.alpaca.markets"

from config import OUTPUT_DIR
from ontology.trigger import get_action_candidates
import trade_journal

RULE_PERF_JSON = OUTPUT_DIR / "rule_performance.json"

# ── 진입 파라미터 (공통) ─────────────────────────────────────────
TRIGGER_MIN_LEVEL = 2       # (vol 폴백용) 2=ALERT, 3=HIGH 만 실행
CONF_REF          = 0.65    # 기준 confidence (여기서 scale=1.0)
CONF_SCALE_CAP    = 2.0     # 강한 신호도 최대 2배까지만
Z_REF             = 2.5     # (vol 폴백용) ALERT 임계 z
MAX_SIGNALS       = 6       # 전략별 한 번에 실행할 최대 시그널 수

# ── 최소 사이즈 모드 (--minimal) ─────────────────────────────────
# 첫 실거래 검증용. trade_journal 채우기 + 호가스프레드/세타 실측이 목적.
# P&L 검증은 표본 누적 후 N≥10 부터 의미.
MINIMAL_BASE_WEIGHT       = 0.005   # 0.5% per signal (vs 통상 3%)
MINIMAL_DIR_BASE_WEIGHT   = 0.005   # 0.5% per signal (vs 통상 5%)
MINIMAL_SHORT_BASE_WEIGHT = 0.003   # 0.3% per signal (숏은 더 보수적)
MINIMAL_MAX_SIGNALS       = 2       # 전략별 최대 2건 (vs 통상 6)

# ── STRADDLE(롱 볼) 파라미터 ─────────────────────────────────────
BASE_WEIGHT       = 0.03    # 시그널당 계좌 자본의 3%를 프리미엄 예산으로
DTE_MIN, DTE_MAX  = 25, 50  # 만기 선택 윈도우 (일)
STRIKE_BAND       = 0.15    # ATM 후보 strike 범위 (±15%)
TAKE_PROFIT_PCT   = 0.25    # 프리미엄 +25% → 익절
STOP_LOSS_PCT     = None    # 하드 손절 없음 — 손실은 DTE로 통제
EXIT_MIN_DTE      = 10      # 만기 10일 이내 → 청산

# ── DIRECTIONAL(현물 롱/숏) 파라미터 ─────────────────────────────
DIR_BASE_WEIGHT   = 0.05    # 시그널당 계좌 자본의 5%를 명목(notional)으로
DIR_TAKE_PROFIT   = 0.10    # 현물 +10% → 익절 (방향 맞음)
DIR_STOP_LOSS     = -0.07   # 현물 -7% → 손절 (방향 틀림)
DIR_MAX_HOLD_DAYS = 21      # 최대 보유 ~1개월: 설명의 예측 지평 안에서 채점

# ── SHORT STRADDLE(숏 볼) 파라미터 (2026-06-10 신설) ────────────
# 시스템 테제 코히어런스 — vol_overpriced ∩ thin_tail_greenlight 만 발화.
# Naked short straddle 무한 손실 위험 → SL +100% (프리미엄 두 배 손실 시 강제 청산).
SHORT_BASE_WEIGHT      = 0.01    # 통상 1% (롱 3% 대비 보수적, naked 위험 반영)
SHORT_TAKE_PROFIT_PCT  = 0.50    # 받은 프리미엄의 50% 회수 시 익절 (vol crush)
SHORT_STOP_LOSS_PCT    = 1.00    # 프리미엄 100% 초과 손실 시 손절 (감마 폭발 방지)
SHORT_EXIT_MIN_DTE     = 10      # 만기 10일 이내 자동 청산 (감마 위험 회피)
SHORT_DTE_MIN, SHORT_DTE_MAX = 25, 50

# signal_type별 사이징 가중. MONITOR(관찰용 의미·광범위 발화)는 소액으로 발주.
# 피드백 승수가 실현성과로 이 prior를 자동 보정.
SIGNAL_TYPE_WEIGHT = {
    "OVERWEIGHT":  1.0,
    "UNDERWEIGHT": 1.0,
    "HEDGE":       1.0,
    "MONITOR":     0.3,
}


def mult_key(rule: str, strategy: str) -> str:
    return f"{rule}|{strategy}"


def load_rule_multipliers() -> dict[str, float]:
    """feedback.py가 생성한 (rule|strategy)별 사이징 승수. 없으면 빈 dict (=모두 1.0)."""
    if not RULE_PERF_JSON.exists():
        return {}
    try:
        data = json.loads(RULE_PERF_JSON.read_text(encoding="utf-8"))
        return {k: float(v.get("multiplier", 1.0)) for k, v in data.items()}
    except (json.JSONDecodeError, AttributeError, TypeError):
        return {}


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


load_env(ROOT / ".env")
TRADE_BASE = os.environ["ALPACA_ENDPOINT"].rstrip("/")
HEADERS = {
    "APCA-API-KEY-ID": os.environ["ALPACA_API_KEY"],
    "APCA-API-SECRET-KEY": os.environ["ALPACA_API_SECRET"],
}


# ── Alpaca 헬퍼 ─────────────────────────────────────────────────
def get_account() -> dict:
    return requests.get(f"{TRADE_BASE}/account", headers=HEADERS, timeout=15).json()


def get_market_open() -> bool:
    try:
        return bool(requests.get(f"{TRADE_BASE}/clock", headers=HEADERS,
                                 timeout=10).json().get("is_open"))
    except Exception:
        return False


def get_positions() -> list[dict]:
    r = requests.get(f"{TRADE_BASE}/positions", headers=HEADERS, timeout=15)
    return r.json() if r.status_code == 200 else []


def get_spot(symbol: str) -> float | None:
    r = requests.get(f"{DATA_BASE}/v2/stocks/{symbol}/trades/latest",
                     headers=HEADERS, timeout=15)
    if r.status_code != 200:
        return None
    return r.json().get("trade", {}).get("p")


def get_contracts(underlying: str, spot: float) -> list[dict]:
    gte = (dt.date.today() + dt.timedelta(days=DTE_MIN)).isoformat()
    lte = (dt.date.today() + dt.timedelta(days=DTE_MAX)).isoformat()
    params = {
        "underlying_symbols": underlying,
        "expiration_date_gte": gte,
        "expiration_date_lte": lte,
        "strike_price_gte": round(spot * (1 - STRIKE_BAND), 2),
        "strike_price_lte": round(spot * (1 + STRIKE_BAND), 2),
        "limit": 1000,
    }
    r = requests.get(f"{TRADE_BASE}/options/contracts", headers=HEADERS,
                     params=params, timeout=20)
    if r.status_code != 200:
        return []
    return r.json().get("option_contracts", [])


def option_mid(symbols: list[str]) -> dict[str, float]:
    """OCC 심볼별 mid price (없으면 ask, 둘 다 없으면 제외)."""
    if not symbols:
        return {}
    r = requests.get(f"{DATA_BASE}/v1beta1/options/quotes/latest",
                     headers=HEADERS, params={"symbols": ",".join(symbols)}, timeout=15)
    out: dict[str, float] = {}
    if r.status_code != 200:
        return out
    for sym, q in r.json().get("quotes", {}).items():
        bid, ask = q.get("bp", 0) or 0, q.get("ap", 0) or 0
        mid = (bid + ask) / 2 if bid and ask else (ask or bid)
        if mid:
            out[sym] = mid
    return out


def submit_option_order(symbol: str, qty: int, side: str = "buy") -> dict:
    body = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": "market",
        "time_in_force": "day",
    }
    r = requests.post(f"{TRADE_BASE}/orders", headers=HEADERS, json=body, timeout=20)
    return {"status_code": r.status_code, "body": r.json()}


def close_position(symbol: str) -> dict:
    """포지션 전량 시장가 청산 (옵션/현물 공통)."""
    r = requests.delete(f"{TRADE_BASE}/positions/{symbol}", headers=HEADERS, timeout=20)
    return {"status_code": r.status_code, "body": r.json()}


def submit_equity_order(symbol: str, qty: int, side: str) -> dict:
    """현물 시장가 주문. side='buy'(롱) / 'sell'(숏). 숏은 마진계좌에서 자동."""
    body = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": "market",
        "time_in_force": "day",
    }
    r = requests.post(f"{TRADE_BASE}/orders", headers=HEADERS, json=body, timeout=20)
    return {"status_code": r.status_code, "body": r.json()}


def held_equity_underlyings() -> set[str]:
    """현재 현물(주식) 포지션이 있는 티커 집합 (방향성 중복 진입 방지용)."""
    out: set[str] = set()
    for p in get_positions():
        if p.get("asset_class") == "us_equity":
            out.add(p["symbol"])
    return out


def parse_occ(sym: str) -> tuple[str, dt.date, str, float]:
    """OCC 심볼 파싱: XLK260710C00180000 → (XLK, 2026-07-10, call, 180.0)."""
    body = sym[-15:]
    underlying = sym[:-15]
    expiry = dt.date(2000 + int(body[0:2]), int(body[2:4]), int(body[4:6]))
    typ = "call" if body[6] == "C" else "put"
    strike = int(body[7:15]) / 1000
    return underlying, expiry, typ, strike


def held_option_underlyings() -> set[str]:
    """현재 옵션 포지션이 있는 기초자산 집합 (중복 진입 방지용)."""
    out: set[str] = set()
    for p in get_positions():
        if p.get("asset_class") == "us_option":
            try:
                out.add(parse_occ(p["symbol"])[0])
            except Exception:
                pass
    return out


# ── 스트래들 진입 계획 ──────────────────────────────────────────
@dataclass
class StraddlePlan:
    ticker: str
    confidence: float
    rule: str
    all_rules: list[str]
    signal_type: str
    regime: str
    conf_multiplier: float
    reasoning: list[str]
    spot: float
    expiry: str
    strike: float
    call_sym: str
    put_sym: str
    call_mid: float
    put_mid: float
    qty: int
    budget: float
    vol_penalty: float = 1.0   # 1.0 미만이면 vol_overpriced 동시 발화 → 사이즈 축소

    @property
    def cost(self) -> float:
        return (self.call_mid + self.put_mid) * 100 * self.qty

    @property
    def over_budget(self) -> bool:
        return self.cost > self.budget * 1.3

    def describe(self) -> str:
        warn = "  ⚠ 예산 초과(최소 1계약)" if self.over_budget else ""
        rules = ", ".join(self.all_rules)
        mult = f"×{self.conf_multiplier:.2f}" if self.conf_multiplier != 1.0 else ""
        type_w = SIGNAL_TYPE_WEIGHT.get(self.signal_type, 1.0)
        type_note = f" (MONITOR ×{type_w})" if type_w != 1.0 else ""
        vol_note = (f" (vol_overpriced ×{self.vol_penalty})"
                    if self.vol_penalty < 1.0 else "")
        return (
            f"  {self.ticker}  conf={self.confidence:.2f}{mult}  [{self.signal_type}]"
            f"{type_note}{vol_note}  spot=${self.spot:.2f}\n"
            f"    트리거 룰: {rules}  (regime={self.regime})\n"
            f"    LONG STRADDLE  {self.expiry}  strike=${self.strike:.0f}  qty={self.qty}\n"
            f"      +CALL {self.call_sym} @ ${self.call_mid:.2f}\n"
            f"      +PUT  {self.put_sym} @ ${self.put_mid:.2f}\n"
            f"      예산 ${self.budget:,.0f}  →  실제비용 ${self.cost:,.0f}{warn}"
        )


def build_plan(cand: dict, equity: float, multipliers: dict[str, float]) -> StraddlePlan | None:
    ticker = cand["ticker"]
    if "=" in ticker or "^" in ticker:      # 선물·지수 제외
        return None

    spot = get_spot(ticker)
    if not spot:
        print(f"  [skip] {ticker}: 현물가 조회 실패")
        return None

    contracts = get_contracts(ticker, spot)
    if not contracts:
        print(f"  [skip] {ticker}: 옵션 계약 없음")
        return None

    expiry = min(c["expiration_date"] for c in contracts)
    leg = [c for c in contracts if c["expiration_date"] == expiry]

    def atm(opt_type: str) -> dict | None:
        cand_legs = [c for c in leg if c["type"] == opt_type]
        return min(cand_legs, key=lambda c: abs(float(c["strike_price"]) - spot)) if cand_legs else None

    call, put = atm("call"), atm("put")
    if not call or not put:
        print(f"  [skip] {ticker}: ATM 콜/풋 부재")
        return None

    mids = option_mid([call["symbol"], put["symbol"]])
    call_mid, put_mid = mids.get(call["symbol"]), mids.get(put["symbol"])
    if not call_mid or not put_mid:
        print(f"  [skip] {ticker}: 옵션 호가 없음 (휴장/유동성)")
        return None

    conf = float(cand.get("confidence", CONF_REF))
    rule = cand.get("rule", "?")
    sig_type = cand.get("signal_type", "?")
    mult = float(multipliers.get(mult_key(rule, "straddle"), 1.0))
    if mult == 0:
        print(f"  [skip] {ticker}: 룰 '{rule}|straddle' stop-rule 발동 (mult=0)")
        return None
    type_w = SIGNAL_TYPE_WEIGHT.get(sig_type, 1.0)
    # 숏 볼 페널티: 같은 사이클에 vol_overpriced 가 발화한 섹터(=IV 비싸다)는
    # 롱 옵션 EV 약화 → trigger.py 가 size_penalty 부착 (기본 1.0)
    vol_pen = float(cand.get("size_penalty", 1.0))
    scale = min(max(conf / CONF_REF, 0.5), CONF_SCALE_CAP) * mult * type_w * vol_pen
    budget = equity * BASE_WEIGHT * scale
    straddle_unit = (call_mid + put_mid) * 100
    qty = max(1, math.floor(budget / straddle_unit))

    return StraddlePlan(
        ticker=ticker, confidence=conf, rule=rule,
        all_rules=cand.get("all_rules", [rule]),
        signal_type=cand.get("signal_type", "?"),
        regime=cand.get("regime", "?"),
        conf_multiplier=mult,
        reasoning=cand.get("reasoning", []),
        spot=spot, expiry=expiry,
        strike=float(call["strike_price"]),
        call_sym=call["symbol"], put_sym=put["symbol"],
        call_mid=call_mid, put_mid=put_mid, qty=qty, budget=budget,
        vol_penalty=vol_pen,
    )


# ── 방향성 진입 계획 (ETF 현물 롱/숏) ───────────────────────────
@dataclass
class DirectionalPlan:
    ticker: str
    direction: int           # +1 롱 / -1 숏
    confidence: float
    rule: str
    all_rules: list[str]
    signal_type: str
    regime: str
    conf_multiplier: float
    reasoning: list[str]
    spot: float
    qty: int
    budget: float

    @property
    def side(self) -> str:
        return "buy" if self.direction > 0 else "sell"

    @property
    def cost(self) -> float:
        return self.spot * self.qty

    def describe(self) -> str:
        arrow = "LONG " if self.direction > 0 else "SHORT"
        rules = ", ".join(self.all_rules)
        mult = f"×{self.conf_multiplier:.2f}" if self.conf_multiplier != 1.0 else ""
        return (
            f"  {self.ticker}  conf={self.confidence:.2f}{mult}  [{self.signal_type}]  spot=${self.spot:.2f}\n"
            f"    트리거 룰: {rules}  (regime={self.regime})\n"
            f"    {arrow} EQUITY  {self.qty}주  명목 ${self.cost:,.0f}  (예산 ${self.budget:,.0f})"
        )


def build_directional_plan(cand: dict, equity: float, multipliers: dict[str, float]) -> DirectionalPlan | None:
    ticker = cand["ticker"]
    if "=" in ticker or "^" in ticker:
        return None

    spot = get_spot(ticker)
    if not spot:
        print(f"  [skip] {ticker}: 현물가 조회 실패")
        return None

    conf = float(cand.get("confidence", CONF_REF))
    rule = cand.get("rule", "?")
    mult = float(multipliers.get(mult_key(rule, "directional"), 1.0))
    if mult == 0:
        print(f"  [skip] {ticker}: 룰 '{rule}|directional' stop-rule 발동 (mult=0)")
        return None
    scale = min(max(conf / CONF_REF, 0.5), CONF_SCALE_CAP) * mult
    budget = equity * DIR_BASE_WEIGHT * scale
    qty = max(1, math.floor(budget / spot))

    return DirectionalPlan(
        ticker=ticker, direction=int(cand.get("direction", 0)), confidence=conf,
        rule=rule, all_rules=cand.get("all_rules", [rule]),
        signal_type=cand.get("signal_type", "?"), regime=cand.get("regime", "?"),
        conf_multiplier=mult, reasoning=cand.get("reasoning", []),
        spot=spot, qty=qty, budget=budget,
    )


# ── 숏 스트래들 진입 계획 (NEW 2026-06-10) ─────────────────────
@dataclass
class ShortStraddlePlan:
    """숏 스트래들 = 콜 매도 + 풋 매도. 받는 프리미엄 (credit). 무한 손실 위험."""
    ticker: str
    confidence: float
    rule: str
    all_rules: list[str]
    signal_type: str
    regime: str
    conf_multiplier: float
    reasoning: list[str]
    spot: float
    expiry: str
    strike: float
    call_sym: str
    put_sym: str
    call_mid: float
    put_mid: float
    qty: int
    budget: float    # 자본 노출 한도 (premium credit 의 부호 반대)

    @property
    def credit(self) -> float:
        """받는 프리미엄 (per share × 100 × qty). 양수."""
        return (self.call_mid + self.put_mid) * 100 * self.qty

    def describe(self) -> str:
        rules = ", ".join(self.all_rules)
        mult = f"×{self.conf_multiplier:.2f}" if self.conf_multiplier != 1.0 else ""
        return (
            f"  {self.ticker}  conf={self.confidence:.2f}{mult}  [{self.signal_type}]"
            f"  spot=${self.spot:.2f}\n"
            f"    트리거 룰: {rules}  (regime={self.regime})\n"
            f"    SHORT STRADDLE  {self.expiry}  strike=${self.strike:.0f}  qty={self.qty}\n"
            f"      -CALL {self.call_sym} @ ${self.call_mid:.2f}\n"
            f"      -PUT  {self.put_sym} @ ${self.put_mid:.2f}\n"
            f"      받는 프리미엄 ${self.credit:,.0f}  "
            f"(SL +{SHORT_STOP_LOSS_PCT*100:.0f}% / TP -{SHORT_TAKE_PROFIT_PCT*100:.0f}%)"
        )


def build_short_straddle_plan(cand: dict, equity: float,
                               multipliers: dict[str, float]
                               ) -> ShortStraddlePlan | None:
    ticker = cand["ticker"]
    if "=" in ticker or "^" in ticker:
        return None

    spot = get_spot(ticker)
    if not spot:
        print(f"  [skip] {ticker}: 현물가 조회 실패")
        return None

    contracts = get_contracts(ticker, spot)
    if not contracts:
        print(f"  [skip] {ticker}: 옵션 계약 없음")
        return None

    expiry = min(c["expiration_date"] for c in contracts)
    leg = [c for c in contracts if c["expiration_date"] == expiry]

    def atm(opt_type: str) -> dict | None:
        cands = [c for c in leg if c["type"] == opt_type]
        return min(cands, key=lambda c: abs(float(c["strike_price"]) - spot)) if cands else None

    call, put = atm("call"), atm("put")
    if not call or not put:
        return None

    mids = option_mid([call["symbol"], put["symbol"]])
    call_mid, put_mid = mids.get(call["symbol"]), mids.get(put["symbol"])
    if not call_mid or not put_mid:
        print(f"  [skip] {ticker}: 옵션 호가 없음")
        return None

    conf = float(cand.get("confidence", CONF_REF))
    rule = cand.get("rule", "?")
    mult = float(multipliers.get(mult_key(rule, "short_straddle"), 1.0))
    if mult == 0:
        print(f"  [skip] {ticker}: 룰 '{rule}|short_straddle' stop-rule (mult=0)")
        return None
    scale = min(max(conf / CONF_REF, 0.5), CONF_SCALE_CAP) * mult
    budget = equity * SHORT_BASE_WEIGHT * scale
    # qty 계산: budget 을 *위험 허용 한도* 로 해석. SL +100% 면 손실 한도 = credit.
    # → qty = budget / (credit_per_unit). credit_per_unit = (call_mid + put_mid) * 100.
    unit_credit = (call_mid + put_mid) * 100
    qty = max(1, math.floor(budget / unit_credit))

    return ShortStraddlePlan(
        ticker=ticker, confidence=conf, rule=rule,
        all_rules=cand.get("all_rules", [rule]),
        signal_type=cand.get("signal_type", "?"),
        regime=cand.get("regime", "?"),
        conf_multiplier=mult, reasoning=cand.get("reasoning", []),
        spot=spot, expiry=expiry,
        strike=float(call["strike_price"]),
        call_sym=call["symbol"], put_sym=put["symbol"],
        call_mid=call_mid, put_mid=put_mid, qty=qty, budget=budget,
    )


# ── 진입 실행 ──────────────────────────────────────────────────
def load_candidates(source: str, json_path: str, run_all_regimes: bool
                    ) -> tuple[list[dict], list[dict], list[dict], str]:
    """트리거 후보 로드 → (롱 스트래들, 방향성, 숏 스트래들, 라벨).

    온톨로지(기본)는 세 전략 후보 모두 반환. vol 폴백은 롱 스트래들만.
    """
    if source == "vol":
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        sigs = [s for s in data.get("etf_signals", [])
                if s.get("level", 0) >= TRIGGER_MIN_LEVEL]
        sigs.sort(key=lambda s: -s.get("z_score", 0))
        cands = []
        for s in sigs:
            z = float(s.get("z_score", Z_REF))
            cands.append({
                "ticker":      s["ticker"],
                "confidence":  min(0.95, 0.55 + (z - Z_REF) * 0.1),
                "rule":        "vol_monitor_z",
                "all_rules":   ["vol_monitor_z"],
                "signal_type": "VOL_SPIKE",
                "regime":      data.get("regime", "?"),
                "reasoning":   [s.get("reason", "")],
            })
        return cands, [], [], f"vol_monitor({data.get('scan_date')})"

    straddle, directional, short_straddle, regime = get_action_candidates(
        run_all_regimes=run_all_regimes)
    return straddle, directional, short_straddle, f"ontology({regime})"


def _refresh_caches() -> None:
    """발주 전 라이브 데이터로 캐시 강제 갱신 (사용자 요청 2026-06-10).

    실패해도 진행 — 마지막 캐시로라도 결정. data_loader 만 yfinance 라이브 호출,
    garch/iv 는 그걸 입력으로 받음. 약 30~60초 소요.
    """
    import subprocess
    here = Path(__file__).resolve().parent
    py = sys.executable
    for script in ("data_loader.py", "garch_estimator.py", "implied_vol.py"):
        try:
            r = subprocess.run([py, str(here / script)], cwd=str(here),
                                capture_output=True, timeout=180)
            tag = "OK" if r.returncode == 0 else f"WARN rc={r.returncode}"
            print(f"  [refresh] {script:25s} {tag}")
        except Exception as exc:
            print(f"  [refresh] {script:25s} FAIL {exc}")


def run_entry(live: bool, json_path: str, source: str = "ontology",
              run_all_regimes: bool = False, minimal: bool = False) -> None:
    global BASE_WEIGHT, DIR_BASE_WEIGHT, SHORT_BASE_WEIGHT  # minimal 임시 override

    if live:
        print("\n[refresh] 라이브 데이터로 캐시 갱신 ...")
        _refresh_caches()

    straddle_c, directional_c, short_straddle_c, src_label = load_candidates(
        source, json_path, run_all_regimes)

    if minimal:
        BASE_WEIGHT = MINIMAL_BASE_WEIGHT
        DIR_BASE_WEIGHT = MINIMAL_DIR_BASE_WEIGHT
        SHORT_BASE_WEIGHT = MINIMAL_SHORT_BASE_WEIGHT
        max_sigs = MINIMAL_MAX_SIGNALS
    else:
        max_sigs = MAX_SIGNALS

    # 보유 섹터 먼저 제외 — 그 다음 상위 N
    held_opt = held_option_underlyings()
    held_eq  = held_equity_underlyings()
    straddle_c       = [c for c in straddle_c       if c["ticker"] not in held_opt][:max_sigs]
    directional_c    = [c for c in directional_c    if c["ticker"] not in held_eq][:max_sigs]
    short_straddle_c = [c for c in short_straddle_c if c["ticker"] not in held_opt][:max_sigs]

    equity = float(get_account().get("equity", 0))
    multipliers = load_rule_multipliers()
    market_open = get_market_open()
    mode = "*** LIVE 주문 제출 ***" if live else "DRY-RUN (주문 미제출)"

    print(f"\n{'='*60}")
    minimal_tag = "  [MINIMAL]" if minimal else ""
    print(f"  진입(ENTRY){minimal_tag}  |  {mode}")
    print(f"  계좌 equity=${equity:,.0f}  시장개장={'예' if market_open else '아니오'}")
    print(f"  트리거: {src_label}  |  롱스트래들 {len(straddle_c)} / "
          f"방향성 {len(directional_c)} / 숏스트래들 {len(short_straddle_c)}")
    if multipliers:
        print(f"  피드백 승수 적용: {len(multipliers)}개 (rule|strategy)")
    print(f"{'='*60}")

    _run_straddle_entry(straddle_c, equity, multipliers, live, market_open)
    _run_directional_entry(directional_c, equity, multipliers, live, market_open)
    _run_short_straddle_entry(short_straddle_c, equity, multipliers, live, market_open)


def _run_straddle_entry(candidates, equity, multipliers, live, market_open) -> None:
    held = held_option_underlyings()
    print(f"\n  ── STRADDLE (롱 볼)  보유중 {sorted(held) or '없음'} ──")

    plans: list[StraddlePlan] = []
    for cand in candidates:
        if cand["ticker"] in held:
            print(f"  [skip] {cand['ticker']}: 이미 옵션 포지션 보유 (중복 진입 방지)")
            continue
        plan = build_plan(cand, equity, multipliers)
        if plan:
            plans.append(plan)
            print(plan.describe())

    if not plans:
        print("  진입할 신규 스트래들 없음.")
        return

    total = sum(p.cost for p in plans)
    print(f"  → {len(plans)}개 스트래들  프리미엄 합계 ${total:,.0f} ({total/equity*100:.1f}% equity)")

    if not live:
        return
    if not market_open:
        print("  [중단] 휴장 중 — 옵션 마켓 주문은 장중에만. (미제출)")
        return

    print("  [LIVE] 스트래들 주문 제출...")
    for p in plans:
        leg_ok = True
        for sym in (p.call_sym, p.put_sym):
            res = submit_option_order(sym, p.qty, side="buy")
            ok = res["status_code"] in (200, 201)
            leg_ok = leg_ok and ok
            oid = res["body"].get("id", res["body"]) if ok else res["body"]
            print(f"    {'OK ' if ok else 'ERR'} buy {p.qty} {sym} -> {oid}")
        if leg_ok:
            tid = trade_journal.log_entry(
                strategy="straddle", ticker=p.ticker, rule=p.rule, all_rules=p.all_rules,
                signal_type=p.signal_type, confidence=p.confidence,
                regime=p.regime, conf_multiplier=p.conf_multiplier,
                spot=p.spot, strike=p.strike, expiry=p.expiry,
                call_sym=p.call_sym, put_sym=p.put_sym, qty=p.qty,
                budget=p.budget, entry_cost=p.cost, reasoning=p.reasoning,
            )
            print(f"    [journal] straddle entry: {tid}")


def _run_directional_entry(candidates, equity, multipliers, live, market_open) -> None:
    held = held_equity_underlyings()
    print(f"\n  ── DIRECTIONAL (현물 롱/숏)  보유중 {sorted(held) or '없음'} ──")

    plans: list[DirectionalPlan] = []
    for cand in candidates:
        if cand["ticker"] in held:
            print(f"  [skip] {cand['ticker']}: 이미 현물 포지션 보유 (중복 진입 방지)")
            continue
        plan = build_directional_plan(cand, equity, multipliers)
        if plan:
            plans.append(plan)
            print(plan.describe())

    if not plans:
        print("  진입할 신규 방향성 포지션 없음.")
        return

    total = sum(p.cost for p in plans)
    print(f"  → {len(plans)}개 방향성  명목 합계 ${total:,.0f} ({total/equity*100:.1f}% equity)")

    if not live:
        return
    if not market_open:
        print("  [중단] 휴장 중 — 현물 주문도 장중에만. (미제출)")
        return

    print("  [LIVE] 방향성 주문 제출...")
    for p in plans:
        res = submit_equity_order(p.ticker, p.qty, side=p.side)
        ok = res["status_code"] in (200, 201)
        oid = res["body"].get("id", res["body"]) if ok else res["body"]
        arrow = "LONG" if p.direction > 0 else "SHORT"
        print(f"    {'OK ' if ok else 'ERR'} {arrow} {p.qty} {p.ticker} -> {oid}")
        if ok:
            tid = trade_journal.log_entry(
                strategy="directional", ticker=p.ticker, rule=p.rule, all_rules=p.all_rules,
                signal_type=p.signal_type, confidence=p.confidence,
                regime=p.regime, conf_multiplier=p.conf_multiplier,
                direction=p.direction, spot=p.spot, entry_price=p.spot,
                qty=p.qty, budget=p.budget, entry_cost=p.cost, reasoning=p.reasoning,
            )
            print(f"    [journal] directional entry: {tid}")


def _run_short_straddle_entry(candidates, equity, multipliers, live, market_open) -> None:
    held = held_option_underlyings()
    plans: list[ShortStraddlePlan] = []
    print(f"\n  ── SHORT STRADDLE (숏 볼)  보유중 {sorted(held)} ──")
    for cand in candidates:
        if cand["ticker"] in held:
            print(f"  [skip] {cand['ticker']}: 옵션 보유 중 (중복 방지)")
            continue
        plan = build_short_straddle_plan(cand, equity, multipliers)
        if plan:
            print(plan.describe())
            plans.append(plan)

    if not plans:
        print("  진입할 신규 숏 스트래들 없음.")
        return

    total_credit = sum(p.credit for p in plans)
    print(f"  → {len(plans)}개 숏 스트래들  받는 프리미엄 합계 ${total_credit:,.0f}")

    if not live:
        return
    if not market_open:
        print("  [중단] 휴장 중 — 옵션 매도도 장중에만. (미제출)")
        return

    print("  [LIVE] 숏 스트래들 주문 제출...")
    for p in plans:
        all_ok = True
        for sym in (p.call_sym, p.put_sym):
            res = submit_option_order(sym, p.qty, side="sell")
            ok = res["status_code"] in (200, 201)
            all_ok = all_ok and ok
            print(f"    {'OK ' if ok else 'ERR'} sell {p.qty} {sym} -> "
                  f"{res['body'].get('id', res['body']) if ok else res['body']}")
        if all_ok:
            tid = trade_journal.log_entry(
                strategy="short_straddle", ticker=p.ticker, rule=p.rule,
                all_rules=p.all_rules, signal_type=p.signal_type,
                confidence=p.confidence, regime=p.regime,
                conf_multiplier=p.conf_multiplier,
                strike=p.strike, expiry=p.expiry,
                call_sym=p.call_sym, put_sym=p.put_sym,
                qty=p.qty, budget=p.budget,
                entry_cost=-p.credit,   # 음의 cost = credit 받음
                reasoning=p.reasoning,
            )
            print(f"    [journal] short_straddle entry: {tid}")


# ── 청산 실행 ──────────────────────────────────────────────────
def run_exit(live: bool) -> None:
    """롱+숏 스트래들(옵션) + 방향성(현물) 청산을 순차 점검.

    옵션 포지션은 long(qty>0) / short(qty<0) 으로 구분해 각각 다른 청산 조건 적용.
    """
    positions = get_positions()
    options = [p for p in positions if p.get("asset_class") == "us_option"]
    long_opts  = [p for p in options if float(p.get("qty", 0)) > 0]
    short_opts = [p for p in options if float(p.get("qty", 0)) < 0]
    _run_straddle_exit(long_opts, live)
    _run_short_straddle_exit(short_opts, live)
    _run_directional_exit(
        [p for p in positions if p.get("asset_class") == "us_equity"], live)


def _run_straddle_exit(positions: list[dict], live: bool) -> None:
    mode = "*** LIVE 청산 ***" if live else "DRY-RUN (청산 미제출)"

    print(f"\n{'='*60}")
    print(f"  스트래들 청산(EXIT)  |  {mode}")
    sl_txt = f"손절 {STOP_LOSS_PCT*100:.0f}%" if STOP_LOSS_PCT is not None else "손절 없음(만기로 통제)"
    print(f"  익절 +{TAKE_PROFIT_PCT*100:.0f}% / {sl_txt} / 만기 {EXIT_MIN_DTE}일 이내")
    print(f"{'='*60}")

    if not positions:
        print("  보유 옵션 포지션 없음.")
        return

    # underlying 별로 묶어 스트래들 단위 평가
    groups: dict[str, list[dict]] = {}
    for p in positions:
        try:
            under = parse_occ(p["symbol"])[0]
        except Exception:
            under = p["symbol"]
        groups.setdefault(under, []).append(p)

    today = dt.date.today()
    to_close: list[tuple[str, list[dict], str, float, float, int]] = []

    for under, legs in groups.items():
        cost = sum(abs(float(l.get("cost_basis", 0) or 0)) for l in legs)
        mv = sum(float(l.get("market_value", 0) or 0) for l in legs)
        pnl_pct = (mv - cost) / cost if cost else 0.0
        min_dte = min((parse_occ(l["symbol"])[1] - today).days for l in legs)

        reason = None
        if pnl_pct >= TAKE_PROFIT_PCT:
            reason = "익절"
        elif STOP_LOSS_PCT is not None and pnl_pct <= STOP_LOSS_PCT:
            reason = "손절"
        elif min_dte <= EXIT_MIN_DTE:
            reason = "만기임박"

        tag = f"→ 청산({reason})" if reason else "유지"
        print(f"  {under}: {len(legs)}레그  비용 ${cost:,.0f}  평가 ${mv:,.0f}  "
              f"P&L {pnl_pct*100:+.1f}%  최소DTE {min_dte}일  {tag}")
        if reason:
            to_close.append((under, legs, reason, mv, pnl_pct, min_dte))

    if not to_close:
        print("\n청산 대상 없음.")
        return

    if not live:
        print(f"\n[DRY-RUN] {len(to_close)}개 스트래들 청산 대상. 실제 청산은 --exit --live.")
        return

    if not get_market_open():
        print("\n[중단] 휴장 중 — 옵션 청산도 장중에만 가능. (청산 미제출)")
        return

    print("\n[LIVE] 청산 주문 제출 중...")
    for under, legs, reason, mv, pnl_pct, min_dte in to_close:
        all_ok = True
        for l in legs:
            res = close_position(l["symbol"])
            ok = res["status_code"] in (200, 201, 207)
            all_ok = all_ok and ok
            print(f"    {'OK ' if ok else 'ERR'} close {l['symbol']} ({reason}) -> {res['status_code']}")
        if all_ok:
            cost = sum(abs(float(l.get("cost_basis", 0) or 0)) for l in legs)
            tid = trade_journal.log_exit(
                strategy="straddle", ticker=under, exit_reason=reason, exit_value=mv,
                pnl_pct=pnl_pct, min_dte=min_dte, entry_cost_fallback=cost,
            )
            print(f"    [journal] exit logged: {tid or '(미매칭)'}")


def _run_short_straddle_exit(positions: list[dict], live: bool) -> None:
    """숏 옵션 포지션(qty < 0) 청산. 받은 프리미엄 기준 +50% 회수 익절,
    +100% 손실(=두 배) 손절, DTE≤10일 강제 청산.

    cost_basis 가 음수 (credit 받음). market_value 도 음수 (도로 갚아야 할 가치).
    pnl = -market_value - (-cost_basis) = cost_basis - market_value
    pnl_pct = pnl / abs(cost_basis)
    """
    mode = "*** LIVE 청산 ***" if live else "DRY-RUN (청산 미제출)"
    print(f"\n{'='*60}")
    print(f"  숏 스트래들 청산  |  {mode}")
    print(f"  익절 -{SHORT_TAKE_PROFIT_PCT*100:.0f}% 회수 / "
          f"손절 +{SHORT_STOP_LOSS_PCT*100:.0f}% 손실 / 만기 {SHORT_EXIT_MIN_DTE}일 이내")
    print(f"{'='*60}")

    if not positions:
        print("  보유 숏 옵션 포지션 없음.")
        return

    # underlying 별 그룹 (콜+풋 페어)
    groups: dict[str, list[dict]] = {}
    for p in positions:
        try:
            under = parse_occ(p["symbol"])[0]
        except Exception:
            under = p["symbol"]
        groups.setdefault(under, []).append(p)

    today = dt.date.today()
    to_close: list[tuple[str, list[dict], str, float, float, int]] = []

    for under, legs in groups.items():
        # cost_basis 음수 (credit 받음). 부호 무시한 절대값 = 받은 프리미엄.
        credit = sum(abs(float(l.get("cost_basis", 0) or 0)) for l in legs)
        # market_value 음수 (갚아야 할 가치). 절대값 = 현재 buyback 비용.
        buyback = sum(abs(float(l.get("market_value", 0) or 0)) for l in legs)
        pnl = credit - buyback   # 양수면 익절 방향
        pnl_pct = pnl / credit if credit else 0.0
        min_dte = min((parse_occ(l["symbol"])[1] - today).days for l in legs)

        reason = None
        if pnl_pct >= SHORT_TAKE_PROFIT_PCT:
            reason = "익절"
        elif pnl_pct <= -SHORT_STOP_LOSS_PCT:
            reason = "손절"
        elif min_dte <= SHORT_EXIT_MIN_DTE:
            reason = "만기임박"

        tag = f"→ 청산({reason})" if reason else "유지"
        print(f"  {under}: {len(legs)}레그  credit ${credit:,.0f}  buyback ${buyback:,.0f}  "
              f"P&L {pnl_pct*100:+.1f}%  최소DTE {min_dte}일  {tag}")
        if reason:
            to_close.append((under, legs, reason, buyback, pnl_pct, min_dte))

    if not to_close:
        print("\n청산 대상 없음.")
        return
    if not live:
        print(f"\n[DRY-RUN] {len(to_close)}개 숏 스트래들 청산 대상.")
        return
    if not get_market_open():
        print("\n[중단] 휴장 중 — 옵션 청산도 장중에만.")
        return

    print("\n[LIVE] 숏 스트래들 청산 (buy_to_close)...")
    for under, legs, reason, buyback, pnl_pct, min_dte in to_close:
        all_ok = True
        for l in legs:
            res = close_position(l["symbol"])
            ok = res["status_code"] in (200, 201, 207)
            all_ok = all_ok and ok
            print(f"    {'OK ' if ok else 'ERR'} buy_to_close {l['symbol']} "
                  f"({reason}) -> {res['status_code']}")
        if all_ok:
            credit = sum(abs(float(l.get("cost_basis", 0) or 0)) for l in legs)
            # exit_value = 받은 credit × (1 + pnl_pct)  (P&L 정합성 위해 환산)
            exit_value = credit * (1 + pnl_pct)
            tid = trade_journal.log_exit(
                strategy="short_straddle", ticker=under, exit_reason=reason,
                exit_value=exit_value, pnl_pct=pnl_pct, min_dte=min_dte,
                entry_cost_fallback=credit,
            )
            print(f"    [journal] short_straddle exit logged: {tid or '(미매칭)'}")


def _run_directional_exit(positions: list[dict], live: bool) -> None:
    """현물(방향성) 청산. 익절/손절은 Alpaca unrealized_plpc, 보유일은 저널 진입ts 기준.

    방향성 베팅은 '설명의 예측 지평'(~수주) 안에서 채점 → 보유일 초과 시 무조건 청산해
    방향이 맞았는지/틀렸는지 실현 P&L로 확정한다 (롱볼 스트래들과 달리 시간을 우호로 두지 않음).
    """
    mode = "*** LIVE 청산 ***" if live else "DRY-RUN (청산 미제출)"

    print(f"\n{'='*60}")
    print(f"  방향성 청산(EXIT)  |  {mode}")
    print(f"  익절 +{DIR_TAKE_PROFIT*100:.0f}% / 손절 {DIR_STOP_LOSS*100:.0f}% / "
          f"최대보유 {DIR_MAX_HOLD_DAYS}일")
    print(f"{'='*60}")

    if not positions:
        print("  보유 현물 포지션 없음.")
        return

    open_dir = {
        tkr: rec for (tkr, strat), rec in trade_journal.open_trades().items()
        if strat == "directional"
    }
    today = dt.date.today()
    to_close: list[tuple[dict, str, float, float]] = []

    for p in positions:
        tkr = p["symbol"]
        mv = float(p.get("market_value", 0) or 0)
        pnl_pct = float(p.get("unrealized_plpc", 0) or 0)
        qty = float(p.get("qty", 0) or 0)

        held_days = None
        entry = open_dir.get(tkr)
        if entry and entry.get("ts"):
            try:
                t0 = dt.datetime.fromisoformat(entry["ts"]).date()
                held_days = (today - t0).days
            except ValueError:
                held_days = None

        reason = None
        if pnl_pct >= DIR_TAKE_PROFIT:
            reason = "익절"
        elif pnl_pct <= DIR_STOP_LOSS:
            reason = "손절"
        elif held_days is not None and held_days >= DIR_MAX_HOLD_DAYS:
            reason = "보유만료"

        side = "LONG " if qty >= 0 else "SHORT"
        held_txt = f"{held_days}일" if held_days is not None else "?일"
        tag = f"→ 청산({reason})" if reason else "유지"
        print(f"  {tkr}: {side} {abs(qty):.0f}주  평가 ${mv:,.0f}  "
              f"P&L {pnl_pct*100:+.1f}%  보유 {held_txt}  {tag}")
        if reason:
            to_close.append((p, reason, mv, pnl_pct))

    if not to_close:
        print("\n청산 대상 없음.")
        return

    if not live:
        print(f"\n[DRY-RUN] {len(to_close)}개 현물 청산 대상. 실제 청산은 --exit --live.")
        return

    if not get_market_open():
        print("\n[중단] 휴장 중 — 현물 청산도 장중에만. (청산 미제출)")
        return

    print("\n[LIVE] 청산 주문 제출 중...")
    for p, reason, mv, pnl_pct in to_close:
        tkr = p["symbol"]
        res = close_position(tkr)
        ok = res["status_code"] in (200, 201, 207)
        print(f"    {'OK ' if ok else 'ERR'} close {tkr} ({reason}) -> {res['status_code']}")
        if ok:
            cost = abs(float(p.get("cost_basis", 0) or 0))
            # 숏은 market_value가 음수 → entry_cost×(1+pnl_pct)로 일관되게 환산
            exit_value = cost * (1 + pnl_pct)
            tid = trade_journal.log_exit(
                strategy="directional", ticker=tkr, exit_reason=reason, exit_value=exit_value,
                pnl_pct=pnl_pct, entry_cost_fallback=cost,
            )
            print(f"    [journal] exit logged: {tid or '(미매칭)'}")


# ── 진입점 ─────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="진입 실제 발주 (장중)")
    ap.add_argument("--exit", action="store_true", dest="do_exit",
                    help="보유 스트래들 청산 점검")
    ap.add_argument("--auto", action="store_true",
                    help="청산→진입 라이브 연속 실행 (스케줄러용, 휴장 시 no-op)")
    ap.add_argument("--source", choices=["ontology", "vol"], default="ontology",
                    help="트리거 소스 (기본 ontology, vol=vol_monitor 폴백)")
    ap.add_argument("--all-regimes", action="store_true", dest="all_regimes",
                    help="온톨로지: 현재 레짐 무관 전체 룰 발화 (리서치 모드)")
    ap.add_argument("--json", default=str(VOL_JSON), help="vol_monitor_data.json 경로 (--source vol)")
    ap.add_argument("--minimal", action="store_true",
                    help="최소 사이즈 모드 (0.5%%/시그널, 최대 2건). 첫 실거래 검증용.")
    args = ap.parse_args()

    if args.auto:
        if not get_market_open():
            print("[auto] 휴장 중 — 진입/청산 건너뜀.")
            return
        run_exit(live=True)
        run_entry(live=True, json_path=args.json,
                  source=args.source, run_all_regimes=args.all_regimes,
                  minimal=args.minimal)
        return

    if args.do_exit:
        run_exit(live=args.live)
        return

    run_entry(live=args.live, json_path=args.json,
              source=args.source, run_all_regimes=args.all_regimes,
              minimal=args.minimal)


if __name__ == "__main__":
    main()

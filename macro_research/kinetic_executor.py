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
STOP_LOSS_PCT     = None    # 하드 손절 없음 — 손실은 DTE로 통제
EXIT_MIN_DTE      = 10      # 만기 10일 이내 → 청산

# 트레일링 청산 (2026-06-12, 라운드 6 비평 1번 — TP +25% 고정 절단 폐지)
# fat-tail 탐지 시스템이 롱 스트래들 이익을 +25%에서 자르는 건 자기부정:
# 위기 컨벡시티 수익 분포는 우측 꼬리가 전부. arm(+25%) 후 피크 이익의
# 40% 반납 시 청산 — 피크 100% 면 60% 에서, 피크 25% 면 15% 에서.
# 사이클 간 피크는 output/trail_state.json 에 유지 (매시간 갱신).
TRAIL_ARM_PCT     = 0.25    # 이 이익률 도달 시 트레일 가동 (구 TP 지점)
TRAIL_GIVEBACK    = 0.40    # 피크 이익률의 40% 반납 → 청산
PARTIAL_FRACTION  = 0.5     # arm 시 부분청산 비율 (계약 ≥2 일 때만, 1회)
TRAIL_STATE_JSON  = OUTPUT_DIR / "trail_state.json"

# ── DIRECTIONAL(현물 롱/숏) 파라미터 ─────────────────────────────
DIR_BASE_WEIGHT   = 0.05    # 시그널당 계좌 자본의 5%를 명목(notional)으로
DIR_TAKE_PROFIT   = 0.10    # 페어 +10% → 익절 (방향 맞음)
DIR_STOP_LOSS     = -0.07   # 페어 -7% → 손절 (방향 틀림)
DIR_MAX_HOLD_DAYS = 21      # 최대 보유 ~1개월: 설명의 예측 지평 안에서 채점

# 베타헤지 페어 (2026-06-12, 라운드 6 비평 2번)
# partial 회귀가 검증한 건 *시장 통제 후* 민감도인데 아웃라이트 숏의 P&L 은
# -β×R_SPY 가 지배 — 검증한 효과와 거래하는 효과가 달랐다. 진입 시 SPY 반대
# 레그(β×명목)를 붙여 페어로 거래: 숏 XLRE + 롱 β·SPY. TP/SL/만료 판정은 페어 P&L.
HEDGE_SYMBOL = "SPY"
PARTIAL_RESULTS_CSV = OUTPUT_DIR / "partial_results.csv"

# 페어 TP/SL = σ_residual 단위 (라운드 7 비평 5번). β 헤지된 잔차 변동성은
# 아웃라이트의 ~1/3 — 고정 ±10/-7% 면 사실상 전부 21일 타임아웃으로 끝난다.
# 21일 보유지평 잔차 σ 기준 ±k σ 로 재정의. 잔차 σ 는 진입 시 저널에 기록.
PAIR_TP_SIGMA = 2.0    # 익절 = +2.0 × σ_residual(21d)
PAIR_SL_SIGMA = 1.5    # 손절 = -1.5 × σ_residual(21d)

# rate 패밀리 노출 캡 (라운드 7 비평 6번). β 헤지로 시장 리스크를 빼면서
# 방향성 북이 순수 금리 베팅 하나로 농축 — 패밀리 합산 섹터레그 명목을 제한.
RATE_FAMILY_CAP_PCT = 0.15
RULE_FAMILY = {"rate_beneficiary": "rate", "rate_victim": "rate"}

# 단일 진실원 (라운드 7 비평 1번): trigger 가 쓰는 (rule × sector) 유효 상태표.
# 룰은퇴(룰 단위)가 못 잡는 *섹터 단위 사망* (예: XLV/XLP 공선성 판명) 을
# '시그널은퇴' 청산으로 회수한다.
SIGNAL_STATE_JSON = OUTPUT_DIR / "rule_sector_state.json"
SIGNAL_STATE_MAX_AGE_H = 48

# event_vol = shadow 격리 (2026-06-30 판정). 룰 발화·SignalNode 기록은 유지, *라이브 진입만* 정지.
# 근거: 닫힌 5건 전부 손실(−$219.5)이 표본 운 아니라 구조적('z 하락=시그널은퇴 청산=vol crush'
# 결합 + 진입 슬리피지) → 메커니즘 판정이라 오염 표본으로도 유효. 게다가 minimal 예산서 clean
# 표본화 구조적 불가(오염 아니면 예산 스킵). ★ 룰셋 9→8 아님(룰 살아있고 발화 기록됨) = 실행
# 격리지 룰 비활성 아님 → OOS 시계 리셋 없음(directional clean_n=1, OOS 6/16 불변).
SHADOW_ENTRY_RULES = {"event_vol"}
EVENT_VOL_SHADOW_LOG = OUTPUT_DIR / "event_vol_shadow_log.jsonl"

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

# ── 예산 초과 가드 (2026-06-12) ─────────────────────────────────
# 최소 1계약(1주) 정책이 confidence 가중 사이징을 역전시키는 것 방지:
# MONITOR ×0.3 으로 예산 $77 인데 1계약이 $1,452 면 의도의 18.8배 노출.
# 1단위 비용이 예산의 BUDGET_OVERRUN_CAP 배를 넘으면 진입 자체를 스킵.
BUDGET_OVERRUN_CAP = 2.0


def mult_key(rule: str, strategy: str) -> str:
    return f"{rule}|{strategy}"


def active_rule_names() -> set[str]:
    """현재 룰셋에 살아있는 룰 short-name 집합.

    inference.ALL_RULES 에서 도출 (rule_natural_hedge → natural_hedge).
    vol_monitor_z 는 --source vol 폴백 트리거라 룰셋 외부 — 항상 유효 취급.
    """
    from ontology.inference import ALL_RULES
    names = {fn.__name__.removeprefix("rule_") for fn in ALL_RULES}
    names.add("vol_monitor_z")
    return names


def load_valid_signal_pairs() -> set[str] | None:
    """단일 진실원 상태표 로드. 없거나 48h 초과 시 None (시그널은퇴 판정 보류)."""
    if not SIGNAL_STATE_JSON.exists():
        return None
    try:
        st = json.loads(SIGNAL_STATE_JSON.read_text(encoding="utf-8"))
        ts = dt.datetime.fromisoformat(st.get("generated", ""))
        if (dt.datetime.now() - ts).total_seconds() > 3600 * SIGNAL_STATE_MAX_AGE_H:
            return None
        return set(st.get("valid_pairs", []))
    except (ValueError, OSError, TypeError):
        return None


def signal_retired_reason(rec: dict | None, valid_pairs: set[str] | None) -> str | None:
    """진입 시그널(rule × sector)이 현재 데이터에서 더 이상 유효하지 않으면 사유.

    룰은퇴(룰 단위)와 별개의 *섹터 단위* 수명주기 — FDR 재판정으로 특정 섹터의
    시그널만 죽는 경우 (XLV/XLP rate_victim 공선성 판명) 를 잡는다.
    상태표 부재/노후 시 None (보수적 보류 — 오청산 방지).
    """
    if not rec or valid_pairs is None:
        return None
    rule, tkr = rec.get("rule"), rec.get("ticker")
    if not rule or not tkr or rule == "vol_monitor_z":
        return None
    if f"{rule}|{tkr}" not in valid_pairs:
        return f"시그널은퇴({rule}|{tkr})"
    return None


def rule_retired_reason(rec: dict | None, strategy: str,
                        active: set[str], multipliers: dict[str, float]
                        ) -> str | None:
    """진입 테제가 시스템에서 폐기됐으면 사유 문자열, 아니면 None.

    두 가지 경로:
      1. 룰이 inference.ALL_RULES 에서 제거됨 (코드 레벨 은퇴)
      2. feedback.py stop-rule 로 multiplier == 0 (성과 레벨 은퇴)
    저널 매칭이 안 되는 포지션(rec=None)은 판정 불가 → 기존 TP/SL/DTE 로직만 적용.
    """
    if not rec:
        return None
    rule = rec.get("rule")
    if not rule:
        return None
    if rule not in active:
        return f"룰은퇴({rule})"
    if multipliers.get(mult_key(rule, strategy)) == 0:
        return f"룰은퇴({rule}|stop-rule)"
    return None


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
    """시장 개장 여부. 2026-06-12: 실패를 휴장으로 *조용히* 위장하지 않는다 —
    GH Actions 러너에서 /clock 실패가 '휴장 no-op'으로 보여 클라우드 kinetic이
    이틀간 무거래였던 원인. 실패 사유를 반드시 출력 (진짜 휴장과 구분)."""
    try:
        r = requests.get(f"{TRADE_BASE}/clock", headers=HEADERS, timeout=10)
        if r.status_code != 200:
            print(f"  [clock] HTTP {r.status_code} {r.text[:120]} — 휴장 취급 (오류!)")
            return False
        return bool(r.json().get("is_open"))
    except Exception as exc:
        print(f"  [clock] 조회 실패 ({exc.__class__.__name__}: {exc}) — 휴장 취급 (오류!)")
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
    if straddle_unit > budget * BUDGET_OVERRUN_CAP:
        print(f"  [skip] {ticker}: 1계약 ${straddle_unit:,.0f} > 예산 ${budget:,.0f}"
              f"×{BUDGET_OVERRUN_CAP:.0f} — 사이징 역전 방지 (최소계약이 의도 노출 초과)")
        return None
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


# ── 방향성 진입 계획 (베타헤지 페어: 섹터 현물 + β·SPY 반대 레그) ──
HEDGE_BETA_WINDOW = 126   # 헤지 베타·잔차 σ 추정 창 (~6개월)


def load_hedge_stats() -> dict[str, dict]:
    """섹터별 {beta, resid_vol(일별)} = 최근 126일 단순 OLS (sec ~ SPY).

    ⚠ partial_results.csv 의 beta_mkt 를 쓰면 안 됨: 그건 VIX·금리 등을 통제한
    *조건부* SPY 계수라 헤지 비율로는 과소 (VIX 변화가 시장 움직임 대부분을
    흡수해 XLP 같은 섹터는 조건부 베타 ≈ 0 으로 나옴). partial 회귀는
    *시그널 검증*, 단순 베타는 *실행 헤지* — 역할이 다르다 (2026-06-12).

    resid_vol = std(sec − β·SPY) 일별 — 페어 TP/SL 의 σ 단위 (라운드 7 비평 5번:
    헤지된 잔차 변동성은 아웃라이트의 ~1/3 라 고정 ±10/-7% 는 사실상 도달 불가).
    """
    path = OUTPUT_DIR / "sector_returns.parquet"
    if not path.exists():
        return {}
    try:
        import pandas as pd
        rets = pd.read_parquet(path).tail(HEDGE_BETA_WINDOW)
        if "SPY" not in rets.columns:
            return {}
        spy = rets["SPY"]
        var = float(spy.var())
        if not (var and math.isfinite(var)):
            return {}
        out: dict[str, dict] = {}
        for col in rets.columns:
            if col == "SPY":
                continue
            b = float(rets[col].cov(spy)) / var
            if not math.isfinite(b):
                continue
            rv = float((rets[col] - b * spy).std())
            out[col] = {"beta": b, "resid_vol": rv if math.isfinite(rv) else 0.0}
        return out
    except Exception:
        return {}


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
    hedge_beta: float = 0.0       # 단순 126D β (실행 헤지용)
    hedge_qty: float = 0.0        # SPY 반대 레그 수량 (소수주, 0 = 헤지 없음)
    hedge_price: float = 0.0      # SPY 현물가 (진입 시점)
    hedge_resid_vol: float = 0.0  # 일별 잔차 σ — 페어 TP/SL 의 단위

    @property
    def side(self) -> str:
        return "buy" if self.direction > 0 else "sell"

    @property
    def hedge_side(self) -> str:
        # 섹터 숏 → SPY 롱 / 섹터 롱 → SPY 숏 (시장 노출 중립화)
        return "buy" if self.direction < 0 else "sell"

    @property
    def cost(self) -> float:
        return self.spot * self.qty

    def describe(self) -> str:
        arrow = "LONG " if self.direction > 0 else "SHORT"
        rules = ", ".join(self.all_rules)
        mult = f"×{self.conf_multiplier:.2f}" if self.conf_multiplier != 1.0 else ""
        hedge_arrow = "LONG" if self.direction < 0 else "SHORT"
        hedge_txt = (f"\n    헤지: {hedge_arrow} {HEDGE_SYMBOL} {self.hedge_qty:.3f}주 "
                     f"(β={self.hedge_beta:.2f}, ${self.hedge_qty * self.hedge_price:,.0f}) "
                     f"— 페어 P&L 로 채점"
                     if self.hedge_qty else
                     "\n    헤지 생략 (β≤0 또는 미미) — 현 레짐 시장 노출 자체가 작음, 아웃라이트")
        return (
            f"  {self.ticker}  conf={self.confidence:.2f}{mult}  [{self.signal_type}]  spot=${self.spot:.2f}\n"
            f"    트리거 룰: {rules}  (regime={self.regime})\n"
            f"    {arrow} EQUITY  {self.qty}주  명목 ${self.cost:,.0f}  (예산 ${self.budget:,.0f})"
            f"{hedge_txt}"
        )


def build_directional_plan(cand: dict, equity: float, multipliers: dict[str, float],
                           hedge_stats: dict[str, dict] | None = None) -> DirectionalPlan | None:
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
    if spot > budget * BUDGET_OVERRUN_CAP:
        print(f"  [skip] {ticker}: 1주 ${spot:,.2f} > 예산 ${budget:,.0f}"
              f"×{BUDGET_OVERRUN_CAP:.0f} — 사이징 역전 방지")
        return None
    qty = max(1, math.floor(budget / spot))

    # 베타헤지 레그 — partial 회귀가 검증한 '시장 대비' 효과를 실제로 거래
    hedge_beta, hedge_qty, hedge_price, hedge_rv = 0.0, 0, 0.0, 0.0
    hs = (hedge_stats or {}).get(ticker) or {}
    beta = hs.get("beta")
    if beta is not None and math.isfinite(beta) and beta > 0:
        spy_spot = get_spot(HEDGE_SYMBOL)
        if spy_spot:
            hedge_notional = beta * qty * spot
            # 소수주 헤지 (Alpaca fractional 지원). floor 절단 시 minimal($500) 명목에선
            # beta<~1.5 가 전부 0 으로 잘려 β헤지(R8 D1)가 무력화됐다 → 소수로 정확히.
            # (섹터숏→SPY롱 매수는 소수 OK. 섹터롱→SPY 소수 공매도는 Alpaca 불가 →
            #  헤지 주문 실패 시 기존 '아웃라이트 폴백' 경로가 처리 — rate_beneficiary 는 현재 KILLED)
            hedge_qty = round(hedge_notional / spy_spot, 3)
            if hedge_qty * spy_spot >= 1.0:   # Alpaca 최소 주문 $1 미만 = dust 스킵
                hedge_beta, hedge_price = beta, spy_spot
                hedge_rv = float(hs.get("resid_vol", 0.0) or 0.0)
            else:
                hedge_qty = 0.0

    return DirectionalPlan(
        ticker=ticker, direction=int(cand.get("direction", 0)), confidence=conf,
        rule=rule, all_rules=cand.get("all_rules", [rule]),
        signal_type=cand.get("signal_type", "?"), regime=cand.get("regime", "?"),
        conf_multiplier=mult, reasoning=cand.get("reasoning", []),
        spot=spot, qty=qty, budget=budget,
        hedge_beta=hedge_beta, hedge_qty=hedge_qty, hedge_price=hedge_price,
        hedge_resid_vol=hedge_rv,
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
    if unit_credit > budget * BUDGET_OVERRUN_CAP:
        print(f"  [skip] {ticker}: 1계약 credit ${unit_credit:,.0f} > 한도 ${budget:,.0f}"
              f"×{BUDGET_OVERRUN_CAP:.0f} — 사이징 역전 방지")
        return None
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
    _reconcile_spy_hedge(live, market_open)   # SPY 풀을 열린 페어 합으로 정합(leak self-heal)
    _run_short_straddle_entry(short_straddle_c, equity, multipliers, live, market_open)
    _reconcile_positions()                    # 저널 미청산 ↔ 라이브 보유 정합(유령 self-heal)


def _log_event_vol_shadow(cand: dict) -> None:
    """event_vol 발화를 별도 로그에 누적(라이브 진입 격리, 6/30 판정). 나중에 instrument
    교체 시 소급 평가용. 사이클당 중복 방지 위해 같은 ticker·같은 날엔 1회만 기록."""
    ticker = cand.get("ticker")
    today = dt.date.today().isoformat()
    if EVENT_VOL_SHADOW_LOG.exists():
        try:
            for line in reversed(EVENT_VOL_SHADOW_LOG.read_text(encoding="utf-8").splitlines()):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("ticker") == ticker and str(r.get("ts", ""))[:10] == today:
                    return   # 오늘 이미 기록됨
        except (OSError, json.JSONDecodeError):
            pass
    rec = {
        "ts":         dt.datetime.now().isoformat(timespec="seconds"),
        "ticker":     ticker,
        "rule":       cand.get("rule"),
        "confidence": cand.get("confidence"),
        "regime":     cand.get("regime"),
        "reasoning":  cand.get("reasoning", []),
        "action":     "shadow_no_entry",
    }
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with EVENT_VOL_SHADOW_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _run_straddle_entry(candidates, equity, multipliers, live, market_open) -> None:
    held = held_option_underlyings()
    print(f"\n  ── STRADDLE (롱 볼)  보유중 {sorted(held) or '없음'} ──")

    plans: list[StraddlePlan] = []
    for cand in candidates:
        if cand.get("rule") in SHADOW_ENTRY_RULES:
            _log_event_vol_shadow(cand)   # 발화 기록 유지, 라이브 진입만 격리(shadow)
            print(f"  [shadow] {cand['ticker']}: {cand.get('rule')} 발화 기록 — "
                  f"라이브 진입 격리(6/30 판정, 진입 0). 발화 누적 로그에 기록.")
            continue
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
    held = held_equity_underlyings() - {HEDGE_SYMBOL}   # SPY 는 헤지 인벤토리 — dedup 제외
    print(f"\n  ── DIRECTIONAL (베타헤지 페어: 섹터 + β·{HEDGE_SYMBOL})  "
          f"보유중 {sorted(held) or '없음'} ──")

    hedge_stats = load_hedge_stats()
    if not hedge_stats:
        print(f"  ⚠ sector_returns.parquet 없음 — 헤지 불가, 아웃라이트 폴백")

    # rate 패밀리 기존 노출 (저널 기준 섹터레그 entry_cost 합)
    family_open: dict[str, float] = {}
    for (tkr, strat), rec in trade_journal.open_trades().items():
        if strat != "directional":
            continue
        fam = RULE_FAMILY.get(rec.get("rule") or "")
        if fam:
            family_open[fam] = family_open.get(fam, 0.0) + abs(float(rec.get("entry_cost") or 0))

    plans: list[DirectionalPlan] = []
    family_planned: dict[str, float] = {}
    for cand in candidates:
        if cand["ticker"] in held:
            print(f"  [skip] {cand['ticker']}: 이미 현물 포지션 보유 (중복 진입 방지)")
            continue
        plan = build_directional_plan(cand, equity, multipliers, hedge_stats)
        if not plan:
            continue
        # 패밀리 노출 캡 — 같은 팩터(rate) 베팅 농축 제한 (라운드 7 비평 6번)
        fam = RULE_FAMILY.get(plan.rule)
        if fam:
            cap = equity * RATE_FAMILY_CAP_PCT
            used = family_open.get(fam, 0.0) + family_planned.get(fam, 0.0)
            if used + plan.cost > cap:
                print(f"  [skip] {plan.ticker}: {fam} 패밀리 캡 — "
                      f"기존 ${used:,.0f} + 신규 ${plan.cost:,.0f} > "
                      f"한도 ${cap:,.0f} ({RATE_FAMILY_CAP_PCT*100:.0f}% equity)")
                continue
            family_planned[fam] = family_planned.get(fam, 0.0) + plan.cost
        plans.append(plan)
        print(plan.describe())

    if not plans:
        print("  진입할 신규 방향성 포지션 없음.")
        return

    total = sum(p.cost for p in plans)
    total_hedge = sum(p.hedge_qty * p.hedge_price for p in plans)
    print(f"  → {len(plans)}개 페어  섹터 명목 ${total:,.0f} + 헤지 명목 ${total_hedge:,.0f} "
          f"({(total+total_hedge)/equity*100:.1f}% equity 그로스)")
    if family_open or family_planned:
        for fam in set(family_open) | set(family_planned):
            print(f"     {fam} 패밀리: 기존 ${family_open.get(fam,0):,.0f} "
                  f"+ 신규 ${family_planned.get(fam,0):,.0f} "
                  f"/ 캡 ${equity*RATE_FAMILY_CAP_PCT:,.0f}")

    if not live:
        return
    if not market_open:
        print("  [중단] 휴장 중 — 현물 주문도 장중에만. (미제출)")
        return

    print("  [LIVE] 페어 주문 제출...")
    for p in plans:
        res = submit_equity_order(p.ticker, p.qty, side=p.side)
        ok = res["status_code"] in (200, 201)
        oid = res["body"].get("id", res["body"]) if ok else res["body"]
        arrow = "LONG" if p.direction > 0 else "SHORT"
        print(f"    {'OK ' if ok else 'ERR'} {arrow} {p.qty} {p.ticker} -> {oid}")
        if not ok:
            continue
        # 헤지 레그 — 섹터 레그 체결 후에만
        hedged = False
        if p.hedge_qty > 0:
            hres = submit_equity_order(HEDGE_SYMBOL, p.hedge_qty, side=p.hedge_side)
            hedged = hres["status_code"] in (200, 201)
            print(f"    {'OK ' if hedged else 'ERR'} hedge {p.hedge_side} "
                  f"{p.hedge_qty:.3f} {HEDGE_SYMBOL} -> "
                  f"{hres['body'].get('id', hres['body']) if hedged else hres['body']}")
            if not hedged:
                print(f"    ⚠ {p.ticker}: 헤지 실패 — 아웃라이트로 저널 기록")
        tid = trade_journal.log_entry(
            strategy="directional", ticker=p.ticker, rule=p.rule, all_rules=p.all_rules,
            signal_type=p.signal_type, confidence=p.confidence,
            regime=p.regime, conf_multiplier=p.conf_multiplier,
            direction=p.direction, spot=p.spot, entry_price=p.spot,
            qty=p.qty, budget=p.budget, entry_cost=p.cost, reasoning=p.reasoning,
            hedge_symbol=HEDGE_SYMBOL if hedged else None,
            hedge_qty=p.hedge_qty if hedged else None,
            hedge_price=p.hedge_price if hedged else None,
            hedge_beta=p.hedge_beta if hedged else None,
            hedge_resid_vol=p.hedge_resid_vol if hedged else None,
        )
        print(f"    [journal] directional entry: {tid}"
              + (f" (β={p.hedge_beta:.2f} 헤지 포함)" if hedged else " (아웃라이트)"))


def _spy_position_qty() -> float:
    """현재 SPY 보유 수량 (롱 +, 숏 -). 없으면 0."""
    for p in get_positions():
        if p.get("symbol") == HEDGE_SYMBOL:
            return float(p.get("qty", 0) or 0)
    return 0.0


def _reconcile_spy_hedge(live: bool, market_open: bool) -> None:
    """SPY 헤지 풀 정합 — 목표(열린 directional 페어 hedge_qty 합, 방향고려) vs 실제 SPY.
    per-pair 매수/되감기가 어긋나면(예: 과거 int 캐스트로 hedge_qty=0 → 청산 시 안 팔려 누적)
    SPY 가 leak 된다. 매 사이클 실제를 목표로 보정해 self-heal (SPY 는 페어 풀이라 합산만 의미)."""
    open_dir = {tkr: rec for (tkr, strat), rec in trade_journal.open_trades().items()
                if strat == "directional"}
    target = 0.0
    for rec in open_dir.values():
        hq = float(rec.get("hedge_qty") or 0)
        if hq:
            sign = +1 if int(rec.get("direction", 0)) < 0 else -1   # 섹터숏→SPY롱(+)
            target += sign * hq
    target = round(target, 3)
    actual = _spy_position_qty()
    diff = round(target - actual, 3)
    spy_spot = get_spot(HEDGE_SYMBOL) or 0.0
    print(f"\n  [SPY 정합] 목표 {target:+.3f}주 / 실제 {actual:+.3f}주 / 차이 {diff:+.3f}주")
    if not spy_spot or abs(diff) * spy_spot < 1.0:
        print("    정합됨 (차이 $1 미만, 보정 불요).")
        return
    side = "buy" if diff > 0 else "sell"
    if not (live and market_open):
        print(f"    [{'DRY-RUN' if not live else '휴장'}] 보정 미제출: {side} {abs(diff):.3f} SPY")
        return
    res = submit_equity_order(HEDGE_SYMBOL, abs(diff), side=side)
    ok = res["status_code"] in (200, 201)
    print(f"    {'OK ' if ok else 'ERR'} SPY 보정 {side} {abs(diff):.3f} -> {res['status_code']}")


def _live_position_counts(positions: list[dict]) -> dict[tuple[str, str], int]:
    """Alpaca 보유 → (ticker, strategy) 별 보유 '세트' 수 (저널 미청산과 대조용).
    옵션 롱(qty≥0)=straddle, 옵션 숏(qty<0)=short_straddle (기초자산별 1세트),
    현물(SPY 헤지 제외)=directional. 스트래들은 콜+풋 2레그라도 1세트로 센다(저널 1진입=1세트)."""
    opt_long: set[str] = set()
    opt_short: set[str] = set()
    eq: set[str] = set()
    for p in positions:
        ac = p.get("asset_class")
        if ac == "us_option":
            try:
                u = parse_occ(p["symbol"])[0]
            except Exception:
                continue
            (opt_long if float(p.get("qty", 0) or 0) >= 0 else opt_short).add(u)
        elif ac == "us_equity" and p.get("symbol") != HEDGE_SYMBOL:
            eq.add(p["symbol"])
    counts: dict[tuple[str, str], int] = {}
    for u in opt_long:
        counts[(u, "straddle")] = 1
    for u in opt_short:
        counts[(u, "short_straddle")] = 1
    for u in eq:
        counts[(u, "directional")] = 1
    return counts


def _reconcile_positions() -> None:
    """저널 미청산 ↔ 라이브 보유 대조 → 유령(고아 진입) 정리. SPY 정합의 '포지션판'.
    회계 정합성만 — 주문 제출 없음(저널만 보정). 라이브/휴장 무관하게 매 사이클 실행 가능
    (Alpaca positions 읽기 + 저널 쓰기만). ★ positions API 실패 시 아무것도 닫지 않음."""
    r = requests.get(f"{TRADE_BASE}/positions", headers=HEADERS, timeout=15)
    positions_ok = r.status_code == 200
    positions = r.json() if positions_ok else []
    if not positions_ok:
        print(f"\n  [포지션 정합] Alpaca positions 조회 실패(HTTP {r.status_code}) "
              f"— 유령 정리 건너뜀(몰살 방지).")
        return
    counts = _live_position_counts(positions)
    cleaned = trade_journal.reconcile_phantoms(counts, positions_ok=True)
    if cleaned:
        for rec in cleaned:
            print(f"\n  [포지션 정합] 유령 정리: {rec.get('ticker')} {rec.get('strategy')} "
                  f"(진입 {str(rec.get('ts',''))[:16]}, cost ${rec.get('entry_cost')}) "
                  f"→ P&L-중립 청산(검증 제외)")
    else:
        print("\n  [포지션 정합] 저널-라이브 정합됨 (유령 없음).")


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


def _load_trail_state() -> dict:
    if not TRAIL_STATE_JSON.exists():
        return {}
    try:
        return json.loads(TRAIL_STATE_JSON.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def _save_trail_state(state: dict) -> None:
    try:
        TRAIL_STATE_JSON.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def _run_straddle_exit(positions: list[dict], live: bool) -> None:
    mode = "*** LIVE 청산 ***" if live else "DRY-RUN (청산 미제출)"

    print(f"\n{'='*60}")
    print(f"  스트래들 청산(EXIT)  |  {mode}")
    print(f"  트레일 arm +{TRAIL_ARM_PCT*100:.0f}% → 피크 {TRAIL_GIVEBACK*100:.0f}% 반납 청산 "
          f"/ 손절 없음(만기 통제) / 만기 {EXIT_MIN_DTE}일 / 룰은퇴 즉시")
    print(f"{'='*60}")

    state = _load_trail_state()

    if not positions:
        print("  보유 옵션 포지션 없음.")
        # 보유가 없으면 잔존 상태 정리
        if state:
            _save_trail_state({})
        return

    # underlying 별로 묶어 스트래들 단위 평가
    groups: dict[str, list[dict]] = {}
    for p in positions:
        try:
            under = parse_occ(p["symbol"])[0]
        except Exception:
            under = p["symbol"]
        groups.setdefault(under, []).append(p)

    # 더 이상 보유하지 않는 underlying 의 트레일 상태 제거
    for stale in [u for u in state if u not in groups]:
        state.pop(stale, None)

    active = active_rule_names()
    multipliers = load_rule_multipliers()
    valid_pairs = load_valid_signal_pairs()
    open_str = {tkr: rec for (tkr, strat), rec in trade_journal.open_trades().items()
                if strat == "straddle"}

    today = dt.date.today()
    to_close: list[tuple[str, list[dict], str, float, float, int]] = []
    to_partial: list[tuple[str, list[dict], float]] = []

    for under, legs in groups.items():
        cost = sum(abs(float(l.get("cost_basis", 0) or 0)) for l in legs)
        mv = sum(float(l.get("market_value", 0) or 0) for l in legs)
        pnl_pct = (mv - cost) / cost if cost else 0.0
        min_dte = min((parse_occ(l["symbol"])[1] - today).days for l in legs)
        retired = (rule_retired_reason(open_str.get(under), "straddle", active, multipliers)
                   or signal_retired_reason(open_str.get(under), valid_pairs))

        st = state.setdefault(under, {"peak_pnl_pct": pnl_pct, "partial_done": False,
                                       "partial_realized": 0.0, "orig_cost": cost})
        if pnl_pct > st.get("peak_pnl_pct", -9.9):
            st["peak_pnl_pct"] = pnl_pct
        peak = st["peak_pnl_pct"]

        # arm 시점 부분청산 (각 레그 qty≥2, 1회) — 이익 일부 고정 + 잔여 무제한
        min_leg_qty = min(abs(float(l.get("qty", 0) or 0)) for l in legs)
        if (not st.get("partial_done") and pnl_pct >= TRAIL_ARM_PCT
                and min_leg_qty >= 2):
            to_partial.append((under, legs, pnl_pct))

        reason = None
        if retired:
            reason = retired   # 폐기된 테제 — 세타 흘리며 대기할 이유 없음
        elif peak >= TRAIL_ARM_PCT and pnl_pct <= peak * (1 - TRAIL_GIVEBACK):
            reason = f"트레일청산(피크{peak*100:+.0f}%)"
        elif STOP_LOSS_PCT is not None and pnl_pct <= STOP_LOSS_PCT:
            reason = "손절"
        elif min_dte <= EXIT_MIN_DTE:
            reason = "만기임박"

        armed = "armed" if peak >= TRAIL_ARM_PCT else "-"
        tag = f"→ 청산({reason})" if reason else "유지"
        print(f"  {under}: {len(legs)}레그  비용 ${cost:,.0f}  평가 ${mv:,.0f}  "
              f"P&L {pnl_pct*100:+.1f}% (피크 {peak*100:+.1f}%, {armed})  "
              f"최소DTE {min_dte}일  {tag}")
        # per-leg 분해 (진단: vol crush vs 방향성 이동 구분 — 콜/풋 손실 비대칭이 단서)
        for l in legs:
            sym = l.get("symbol", "")
            lt = "CALL" if (len(sym) >= 9 and sym[-9] == "C") else "PUT "
            lc = abs(float(l.get("cost_basis", 0) or 0))
            lmv = float(l.get("market_value", 0) or 0)
            lpnl = (lmv - lc) / lc * 100 if lc else 0.0
            print(f"      └ {lt} {sym[-9:]}  ${lc:,.0f}→${lmv:,.0f} ({lpnl:+.0f}%)")
        if reason:
            to_close.append((under, legs, reason, mv, pnl_pct, min_dte))

    _save_trail_state(state)

    if to_partial and not live:
        print(f"\n[DRY-RUN] 부분익절 대상 {len(to_partial)}건 "
              f"(arm 도달, 레그당 {PARTIAL_FRACTION*100:.0f}% 매도): "
              + ", ".join(u for u, _, _ in to_partial))

    if not to_close and not (to_partial and live):
        if not to_close:
            print("\n청산 대상 없음.")
        if not live:
            return

    if not live:
        print(f"\n[DRY-RUN] {len(to_close)}개 스트래들 청산 대상. 실제 청산은 --exit --live.")
        return

    if (to_close or to_partial) and not get_market_open():
        print("\n[중단] 휴장 중 — 옵션 청산도 장중에만 가능. (미제출)")
        return

    # 부분익절 — 각 레그 절반 매도 (실현분을 상태에 기록, 최종 저널에 합산)
    for under, legs, pnl_pct in to_partial:
        st = state.get(under, {})
        all_ok = True
        realized = 0.0
        for l in legs:
            qty = int(abs(float(l.get("qty", 0) or 0)))
            qty_close = max(1, int(qty * PARTIAL_FRACTION))
            res = submit_option_order(l["symbol"], qty_close, side="sell")
            ok = res["status_code"] in (200, 201)
            all_ok = all_ok and ok
            if ok:
                leg_cost = abs(float(l.get("cost_basis", 0) or 0))
                leg_mv = float(l.get("market_value", 0) or 0)
                realized += (leg_mv - leg_cost) * (qty_close / qty)
            print(f"    {'OK ' if ok else 'ERR'} 부분익절 sell {qty_close} {l['symbol']} "
                  f"-> {res['status_code']}")
        if all_ok:
            st["partial_done"] = True
            st["partial_realized"] = round(st.get("partial_realized", 0.0) + realized, 2)
            print(f"    [trail] {under} 부분익절 실현 ${realized:+,.0f} — 잔여분 트레일 지속")
    _save_trail_state(state)

    if not to_close:
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
            st = state.pop(under, {})
            cost_rem = sum(abs(float(l.get("cost_basis", 0) or 0)) for l in legs)
            partial = float(st.get("partial_realized", 0.0) or 0.0)
            orig_cost = float(st.get("orig_cost", cost_rem) or cost_rem)
            total_pnl = (mv - cost_rem) + partial
            tid = trade_journal.log_exit(
                strategy="straddle", ticker=under, exit_reason=reason,
                exit_value=orig_cost + total_pnl,
                pnl_pct=(total_pnl / orig_cost if orig_cost else pnl_pct),
                min_dte=min_dte, entry_cost_fallback=orig_cost,
            )
            print(f"    [journal] exit logged: {tid or '(미매칭)'}"
                  + (f"  (부분익절 ${partial:+,.0f} 합산)" if partial else ""))
    _save_trail_state(state)


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
          f"손절 +{SHORT_STOP_LOSS_PCT*100:.0f}% 손실 / 만기 {SHORT_EXIT_MIN_DTE}일 이내 / 룰은퇴 즉시")
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

    active = active_rule_names()
    multipliers = load_rule_multipliers()
    valid_pairs = load_valid_signal_pairs()
    open_short = {tkr: rec for (tkr, strat), rec in trade_journal.open_trades().items()
                  if strat == "short_straddle"}

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
        retired = (rule_retired_reason(open_short.get(under), "short_straddle",
                                       active, multipliers)
                   or signal_retired_reason(open_short.get(under), valid_pairs))

        reason = None
        if pnl_pct >= SHORT_TAKE_PROFIT_PCT:
            reason = "익절"
        elif retired:
            reason = retired   # naked 숏 — 폐기된 테제를 감마 리스크로 끌고 갈 이유 없음
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


def _pair_pnl_pct(p: dict, entry: dict | None, spy_now: float | None) -> tuple[float, bool]:
    """(pnl_pct, is_pair). 헤지 저널이 있으면 페어 P&L (검증된 시장-대비 효과),
    없으면 레거시 아웃라이트 plpc."""
    outright = float(p.get("unrealized_plpc", 0) or 0)
    if not entry or not entry.get("hedge_qty") or not spy_now:
        return outright, False
    cost = abs(float(p.get("cost_basis", 0) or 0))
    if not cost:
        return outright, False
    sector_pnl = cost * outright
    h_qty   = float(entry["hedge_qty"])
    h_price = float(entry.get("hedge_price") or 0)
    h_dir   = +1 if int(entry.get("direction", 0)) < 0 else -1   # 섹터숏→SPY롱
    hedge_pnl = h_qty * (spy_now - h_price) * h_dir if h_price else 0.0
    return (sector_pnl + hedge_pnl) / cost, True


def _run_directional_exit(positions: list[dict], live: bool) -> None:
    """방향성(베타헤지 페어) 청산. 판정은 *페어 P&L* — partial 회귀가 검증한
    시장-대비 효과 그 자체로 채점 (2026-06-12). 레거시 무헤지 포지션은 아웃라이트.

    보유일 초과 시 무조건 청산해 방향이 맞았는지 실현 P&L로 확정한다.
    SPY 포지션은 여러 페어의 헤지 인벤토리 합산이므로 직접 평가하지 않고,
    각 페어 청산 시 해당 hedge_qty 만큼만 반대 주문으로 되감는다.
    """
    mode = "*** LIVE 청산 ***" if live else "DRY-RUN (청산 미제출)"

    print(f"\n{'='*60}")
    print(f"  방향성 청산(EXIT — 페어 P&L 기준)  |  {mode}")
    print(f"  익절 +{DIR_TAKE_PROFIT*100:.0f}% / 손절 {DIR_STOP_LOSS*100:.0f}% / "
          f"최대보유 {DIR_MAX_HOLD_DAYS}일 / 룰은퇴 즉시")
    print(f"{'='*60}")

    positions = [p for p in positions if p["symbol"] != HEDGE_SYMBOL]
    if not positions:
        print("  보유 현물 포지션 없음 (헤지 인벤토리 제외).")
        return

    open_dir = {
        tkr: rec for (tkr, strat), rec in trade_journal.open_trades().items()
        if strat == "directional"
    }
    needs_spy = any(e.get("hedge_qty") for e in open_dir.values())
    spy_now = get_spot(HEDGE_SYMBOL) if needs_spy else None

    active = active_rule_names()
    multipliers = load_rule_multipliers()
    valid_pairs = load_valid_signal_pairs()
    hedge_stats = load_hedge_stats()   # 재페어화 판정용 (β>0 이어야 헤지 붙음)
    today = dt.date.today()
    to_close: list[tuple[dict, str, float, float, dict | None]] = []

    for p in positions:
        tkr = p["symbol"]
        mv = float(p.get("market_value", 0) or 0)
        qty = float(p.get("qty", 0) or 0)
        entry = open_dir.get(tkr)
        pnl_pct, is_pair = _pair_pnl_pct(p, entry, spy_now)

        held_days = None
        if entry and entry.get("ts"):
            try:
                t0 = dt.datetime.fromisoformat(entry["ts"]).date()
                held_days = (today - t0).days
            except ValueError:
                held_days = None

        retired = (rule_retired_reason(entry, "directional", active, multipliers)
                   or signal_retired_reason(entry, valid_pairs))

        # TP/SL 임계 — 페어는 σ_residual(21d) 단위 (라운드 7 비평 5번:
        # 헤지 잔차 변동성은 아웃라이트의 ~1/3 라 고정 ±10/-7% 는 도달 불가)
        tp, sl = DIR_TAKE_PROFIT, DIR_STOP_LOSS
        rv = float(entry.get("hedge_resid_vol") or 0) if entry else 0.0
        if is_pair and rv > 0:
            sig_hold = rv * math.sqrt(DIR_MAX_HOLD_DAYS)
            tp = PAIR_TP_SIGMA * sig_hold
            sl = -PAIR_SL_SIGMA * sig_hold

        reason = None
        if pnl_pct >= tp:
            reason = "익절"
        elif retired:
            reason = retired
        elif pnl_pct <= sl:
            reason = "손절"
        elif (entry and not entry.get("hedge_qty") and not is_pair
              and valid_pairs is not None
              and f"{entry.get('rule')}|{tkr}" in valid_pairs
              and float((hedge_stats.get(tkr) or {}).get("beta") or 0) > 0):
            # floor 버그로 아웃라이트로 들어간 포지션 — 신호 유효 + 헤지 붙음(β>0)이면
            # 청산 → 다음 사이클에 소수주 β헤지 페어로 재진입 (R8 D1 정합화, 2026-06-16).
            reason = "구조전환(페어화)"
        elif held_days is not None and held_days >= DIR_MAX_HOLD_DAYS:
            reason = "보유만료"

        side = "LONG " if qty >= 0 else "SHORT"
        held_txt = f"{held_days}일" if held_days is not None else "?일"
        pair_txt = (f"페어 TP{tp*100:+.1f}/SL{sl*100:+.1f}%" if (is_pair and rv > 0)
                    else ("페어(σ無→고정임계)" if is_pair else "아웃라이트"))
        tag = f"→ 청산({reason})" if reason else "유지"
        print(f"  {tkr}: {side} {abs(qty):.0f}주  평가 ${mv:,.0f}  "
              f"P&L {pnl_pct*100:+.1f}% ({pair_txt})  보유 {held_txt}  {tag}")
        if reason:
            to_close.append((p, reason, mv, pnl_pct, entry))

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
    for p, reason, mv, pnl_pct, entry in to_close:
        tkr = p["symbol"]
        res = close_position(tkr)
        ok = res["status_code"] in (200, 201, 207)
        print(f"    {'OK ' if ok else 'ERR'} close {tkr} ({reason}) -> {res['status_code']}")
        if not ok:
            continue
        # 헤지 되감기 — SPY 전체가 아니라 이 페어의 hedge_qty 만 반대 주문
        if entry and entry.get("hedge_qty"):
            h_qty = float(entry["hedge_qty"])
            unwind_side = "sell" if int(entry.get("direction", 0)) < 0 else "buy"
            hres = submit_equity_order(HEDGE_SYMBOL, h_qty, side=unwind_side)
            hok = hres["status_code"] in (200, 201)
            print(f"    {'OK ' if hok else 'ERR'} unwind hedge {unwind_side} "
                  f"{h_qty:.3f} {HEDGE_SYMBOL} -> {hres['status_code']}")
        cost = abs(float(p.get("cost_basis", 0) or 0))
        # pnl_pct 는 페어 기준 — exit_value 도 페어 P&L 로 환산
        exit_value = cost * (1 + pnl_pct)
        tid = trade_journal.log_exit(
            strategy="directional", ticker=tkr, exit_reason=reason, exit_value=exit_value,
            pnl_pct=pnl_pct, entry_cost_fallback=cost,
        )
        print(f"    [journal] exit logged: {tid or '(미매칭)'}")


# ── 진입점 ─────────────────────────────────────────────────────
CHURN_FILLS_JSON = OUTPUT_DIR / "churn_fills.json"
CHURN_SINCE = "2026-06-16"   # freeze v2 = 전향 시작. 이후 체결만 churn 집계.
OPT_MULTIPLIER = 100         # 옵션 1계약 = 100주


def account_activities(activity_type: str, after: str = CHURN_SINCE,
                       page_size: int = 100) -> list[dict]:
    """Alpaca account activities (체결=FILL 등). after(YYYY-MM-DD) 이후, page_token 페이지네이션.
    실패/비200 이면 빈 리스트 (호출측에서 fill 기반 산출 스킵 → 저널 폴백)."""
    out: list[dict] = []
    page_token = None
    for _ in range(50):   # 안전 상한 (5000건)
        params: dict = {"page_size": page_size, "after": after}
        if page_token:
            params["page_token"] = page_token
        r = requests.get(f"{TRADE_BASE}/account/activities/{activity_type}",
                         headers=HEADERS, params=params, timeout=20)
        if r.status_code != 200:
            break
        batch = r.json()
        if not batch:
            break
        out.extend(batch)
        if len(batch) < page_size:
            break
        page_token = batch[-1].get("id")
    return out


def compute_churn_from_fills(since: str = CHURN_SINCE) -> dict | None:
    """Alpaca 체결 이력 → '같은날 옵션 왕복'(=churn 깜빡임)의 실거래가 실현손익.

    저널 mid-pnl 은 (a)mid 추정 (b)키-충돌로 엉뚱한 진입 매칭 → churn 비용 과소(2.7배).
    여기선 *실체결가*로 심볼별 FIFO 매칭, 같은날 열렸다 닫힌 로트만 churn 으로 집계.
    옵션만 대상(주식 fill 제외). 실패 시 None → 호출측 저널 폴백.
    """
    fills = account_activities("FILL", after=since)
    if not fills:
        return None
    from collections import defaultdict
    legs: dict[str, list[dict]] = defaultdict(list)
    for f in fills:
        sym = f.get("symbol", "")
        try:
            parse_occ(sym)   # 옵션만 (실패=주식 → 제외)
        except Exception:
            continue
        legs[sym].append({
            "t":     str(f.get("transaction_time", "")),
            "side":  f.get("side", ""),
            "qty":   abs(float(f.get("qty", 0) or 0)),
            "price": float(f.get("price", 0) or 0),
        })

    # 심볼별 FIFO: buy 로트 적재, sell 로 닫음. 같은날 닫힌 로트 = churn 왕복 레그.
    by_ud: dict[tuple, dict] = defaultdict(lambda: {"realized": 0.0, "legs": 0})
    for sym, fs in legs.items():
        under = parse_occ(sym)[0]
        fs.sort(key=lambda x: x["t"])
        lots: list[list] = []   # [open_day, price, qty_remaining]
        for f in fs:
            if f["side"] == "buy":
                lots.append([f["t"][:10], f["price"], f["qty"]])
            elif f["side"] in ("sell", "sell_short"):
                qrem = f["qty"]
                while qrem > 1e-9 and lots:
                    lot = lots[0]
                    take = min(qrem, lot[2])
                    if lot[0] == f["t"][:10]:   # 같은날 왕복 = churn
                        realized = (f["price"] - lot[1]) * take * OPT_MULTIPLIER
                        k = (under, f["t"][:10])
                        by_ud[k]["realized"] += realized
                        by_ud[k]["legs"] += 1
                    lot[2] -= take
                    qrem -= take
                    if lot[2] <= 1e-9:
                        lots.pop(0)

    cum_pnl = round(sum(v["realized"] for v in by_ud.values()), 2)
    flickers = sum(v["legs"] // 2 for v in by_ud.values())   # 콜+풋 한쌍 = 1 깜빡임
    detail = [{"underlying": k[0], "day": k[1],
               "roundtrips": v["legs"] // 2, "realized": round(v["realized"], 2)}
              for k, v in sorted(by_ud.items())]
    return {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "basis":     "alpaca_fills",
        "since":     since,
        "flickers":  flickers,
        "cum_pnl":   cum_pnl,
        "by_underlying_day": detail,
    }


def write_churn_fills() -> None:
    """churn(fill 기반) 산출 → output/churn_fills.json. attribution 이 읽어 forward_validation 에 반영.
    읽기 전용(주문 없음)·실패 무해(저널 폴백). 매 사이클(휴장 포함) 호출."""
    try:
        res = compute_churn_from_fills()
        if res is None:
            print("  [churn-fills] 체결 이력 조회 실패/없음 — 저널 폴백 유지.")
            return
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        CHURN_FILLS_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
        print(f"  [churn-fills] flickers={res['flickers']} cum_pnl=${res['cum_pnl']} "
              f"(실거래가 기반) → churn_fills.json")
    except Exception as exc:
        print(f"  [churn-fills] 산출 스킵: {exc}")


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

    write_churn_fills()   # churn 비용 fill 기반 산출 (읽기 전용, 휴장 무관·매 사이클)

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

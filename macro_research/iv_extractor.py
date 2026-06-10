"""
iv_extractor.py — Alpaca 옵션 체인에서 섹터별 ATM IV 추출 (진짜 VRP 계산용).

배경
─────
기존 `garch_estimator.compute_vrp` 는 VRP = σ_GARCH - RV_20d 로 계산했는데,
σ_GARCH 는 *물리측도(P)* 의 모델 예측이지 *위험중립측도(Q)* 의 IV 가 아니다.
variance risk premium 의 정의는 IV(Q) − RV(P) — "옵션이 받아내는 값 vs 실제 실현될 값"
의 갭이다. σ_GARCH 와 RV 의 차이는 변동성 모멘텀/평균회귀 신호일 뿐 옵션 가격
정보가 0% 들어가 있지 않다.

본 모듈은 Alpaca 옵션 체인에서 *진짜 IV* 를 추출한다:
1. ATM 콜+풋 (DTE 20~50) 의 mid 가격을 받음 (kinetic_executor 의 헬퍼 재사용)
2. Alpaca snapshot endpoint 가 `implied_volatility` 를 직접 제공하면 그 값을 사용
3. 미제공이면 Black-Scholes 역산 (scipy.optimize.brentq)
4. ATM 콜 IV + ATM 풋 IV 의 평균 = 그 섹터의 ATM IV (연환산)

산출: dict[ticker, dict] — {'iv': float, 'expiry': str, 'strike': float, 'dte': int,
                              'source': 'snapshot' | 'bs_inverse'}
주의 사항
────────
- 휴장 중엔 quotes 비어 NaN 반환 가능. 호출자는 fallback 로직 필요.
- IV 역산은 만기까지의 시간을 영업일/달력일 중 *달력일* 로 (DTE/365).
- 무위험금리는 단기물(3개월 T-bill) 근사 — 환경변수 RISK_FREE_RATE 우선,
  미설정 시 0.05 (= 5%) 폴백.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

# kinetic_executor 의 Alpaca 헬퍼 재사용 — 중복 정의 회피
from kinetic_executor import (  # type: ignore
    DATA_BASE, HEADERS, TRADE_BASE,
    get_spot, get_contracts, parse_occ,
)
from config import OUTPUT_DIR

RISK_FREE_RATE = float(os.environ.get("RISK_FREE_RATE", "0.05"))
DTE_MIN, DTE_MAX = 20, 50

# 매시간 cron 이 한 행씩 append. parquet 보다 동시성·재시작에 안전.
# 1주 후 11 섹터 × 24 × 7 ≈ 1,848 행 — 분석 시 pandas 로 한 번에 로드.
HISTORY_FILE = OUTPUT_DIR / "iv_history.jsonl"
# 사이트 동기화 대상: macro-portal/public/data 에도 함께 발행 (UI 차트용)
PORTAL_DATA_DIR = OUTPUT_DIR.parent.parent / "macro-portal" / "public" / "data"


# ---------------------------------------------------------------------------
# Black-Scholes implied vol (역산)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_price(S: float, K: float, T: float, r: float, sigma: float,
              opt_type: str) -> float:
    """무배당 가정 Black-Scholes 가격."""
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if opt_type == "call" else (K - S))
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt_type == "call":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def implied_vol(price: float, S: float, K: float, T: float, r: float,
                opt_type: str) -> float | None:
    """Brent 이분법으로 BS implied vol 역산. 실패 시 None."""
    if price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None
    # 가격이 내재가치 이하면 IV 미정의
    intrinsic = max(0.0, (S - K) if opt_type == "call" else (K - S))
    if price < intrinsic - 1e-4:
        return None

    def f(sigma: float) -> float:
        return _bs_price(S, K, T, r, sigma, opt_type) - price

    lo, hi = 1e-4, 5.0
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        return None
    # 50회 이분법
    for _ in range(60):
        mid = (lo + hi) / 2
        f_mid = f(mid)
        if abs(f_mid) < 1e-6:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Alpaca 옵션 데이터 (snapshot endpoint 우선)
# ---------------------------------------------------------------------------

def _snapshot(symbols: list[str]) -> dict[str, dict]:
    """Alpaca options snapshot — 한 번에 여러 심볼 받음. {sym: {iv, mid, ...}}."""
    if not symbols:
        return {}
    r = requests.get(f"{DATA_BASE}/v1beta1/options/snapshots",
                     headers=HEADERS,
                     params={"symbols": ",".join(symbols)}, timeout=15)
    if r.status_code != 200:
        return {}
    out: dict[str, dict] = {}
    for sym, snap in r.json().get("snapshots", {}).items():
        q = snap.get("latestQuote") or {}
        bid, ask = q.get("bp", 0) or 0, q.get("ap", 0) or 0
        mid = (bid + ask) / 2 if bid and ask else (ask or bid)
        iv = snap.get("impliedVolatility")
        out[sym] = {"mid": mid, "iv": iv}
    return out


# ---------------------------------------------------------------------------
# ATM IV per ticker
# ---------------------------------------------------------------------------

def get_atm_iv(ticker: str, spot: float | None = None) -> dict | None:
    """티커의 ATM 콜+풋 평균 IV (연환산) 반환.

    Returns: {'iv', 'iv_call', 'iv_put', 'expiry', 'strike', 'dte', 'source'} 또는 None.
    """
    if spot is None:
        spot = get_spot(ticker)
    if not spot:
        return None

    contracts = get_contracts(ticker, spot)
    if not contracts:
        return None

    # DTE 범위 안에서 가장 가까운 만기
    today = dt.date.today()
    def _dte(c: dict) -> int:
        try:
            d = dt.date.fromisoformat(c["expiration_date"])
            return (d - today).days
        except (ValueError, KeyError):
            return 9999
    contracts = [c for c in contracts if DTE_MIN <= _dte(c) <= DTE_MAX]
    if not contracts:
        return None
    expiry = min(c["expiration_date"] for c in contracts)
    leg = [c for c in contracts if c["expiration_date"] == expiry]

    def _atm(opt_type: str) -> dict | None:
        cands = [c for c in leg if c["type"] == opt_type]
        return min(cands, key=lambda c: abs(float(c["strike_price"]) - spot)) \
            if cands else None

    call, put = _atm("call"), _atm("put")
    if not call or not put:
        return None

    snaps = _snapshot([call["symbol"], put["symbol"]])
    iv_call = (snaps.get(call["symbol"]) or {}).get("iv")
    iv_put  = (snaps.get(put["symbol"])  or {}).get("iv")
    source = "snapshot"

    # snapshot 가 IV 미제공이면 BS 역산
    K = float(call["strike_price"])
    dte = (dt.date.fromisoformat(expiry) - today).days
    T = max(dte / 365.0, 1e-6)

    if iv_call is None or iv_put is None:
        c_mid = (snaps.get(call["symbol"]) or {}).get("mid")
        p_mid = (snaps.get(put["symbol"])  or {}).get("mid")
        if not c_mid or not p_mid:
            return None
        iv_call = implied_vol(c_mid, spot, K, T, RISK_FREE_RATE, "call") or iv_call
        iv_put  = implied_vol(p_mid, spot, K, T, RISK_FREE_RATE, "put")  or iv_put
        source = "bs_inverse"

    if iv_call is None or iv_put is None:
        return None

    iv_avg = (float(iv_call) + float(iv_put)) / 2.0
    return {
        "iv":      round(iv_avg, 4),
        "iv_call": round(float(iv_call), 4),
        "iv_put":  round(float(iv_put), 4),
        "expiry":  expiry,
        "strike":  K,
        "dte":     dte,
        "source":  source,
    }


# ---------------------------------------------------------------------------
# History (append-only JSONL — 매시간 cron 누적)
# ---------------------------------------------------------------------------

def append_history(ticker: str, info: dict, rv_20d: float,
                    vrp_iv: float) -> None:
    """한 행 append. 매시간 populate 가 호출."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts":      dt.datetime.now().isoformat(timespec="seconds"),
        "ticker":  ticker,
        "iv_atm":  info["iv"],
        "iv_call": info["iv_call"],
        "iv_put":  info["iv_put"],
        "strike":  info["strike"],
        "expiry":  info["expiry"],
        "dte":     info["dte"],
        "source":  info["source"],
        "rv_20d":  round(float(rv_20d), 4),
        "vrp_iv":  round(float(vrp_iv), 4),
    }
    with HISTORY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # 사이트 동기화 — 정적 JSON 으로 변환 (마지막 7일 / 168시간 = 차트용)
    _sync_to_portal()


def _sync_to_portal() -> None:
    """history.jsonl → macro-portal/public/data/iv_history.json (UI fetch 대상)."""
    if not PORTAL_DATA_DIR.exists():
        return
    if not HISTORY_FILE.exists():
        return
    try:
        records = []
        for line in HISTORY_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        # 최근 7일치만 (RTH 동안 매시간 × 11 섹터 = ~1,848)
        cutoff = (dt.datetime.now() - dt.timedelta(days=7)).isoformat()
        recent = [r for r in records if r.get("ts", "") >= cutoff]
        payload = {
            "generated": dt.datetime.now().isoformat(timespec="seconds"),
            "n_records": len(recent),
            "records": recent,
        }
        (PORTAL_DATA_DIR / "iv_history.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass   # 동기화 실패는 메인 흐름 영향 안 줌


def load_history():
    """전체 history → pandas DataFrame (없으면 빈 DF)."""
    import pandas as pd
    if not HISTORY_FILE.exists():
        return pd.DataFrame()
    df = pd.read_json(HISTORY_FILE, lines=True, convert_dates=["ts"])
    return df.sort_values(["ticker", "ts"]).reset_index(drop=True)


def summarize_recent(days: int = 7) -> "pd.DataFrame":
    """최근 N일 섹터별 VRP_true 분포 요약 (cron 누적 후 분석용)."""
    import pandas as pd
    df = load_history()
    if df.empty:
        return df
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    df = df[df["ts"] >= cutoff]
    return df.groupby("ticker").agg(
        n_obs=("vrp_iv", "size"),
        vrp_iv_mean=("vrp_iv", "mean"),
        vrp_iv_std=("vrp_iv", "std"),
        vrp_iv_latest=("vrp_iv", "last"),
        iv_atm_latest=("iv_atm", "last"),
    ).round(4)


# ---------------------------------------------------------------------------
# CLI diagnostic
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from config import SECTOR_ETFS
    print(f"[iv_extractor] ATM IV 추출 (DTE {DTE_MIN}~{DTE_MAX}, "
          f"r={RISK_FREE_RATE:.2%}) ...\n")
    print(f"  {'ticker':6s} {'iv':>7s} {'call':>7s} {'put':>7s} "
          f"{'dte':>4s} {'K':>7s}  source")
    print(f"  {'-'*60}")
    for tkr in SECTOR_ETFS:
        info = get_atm_iv(tkr)
        if not info:
            print(f"  {tkr:6s} (skip - data 없음)")
            continue
        print(f"  {tkr:6s} {info['iv']*100:>6.1f}% {info['iv_call']*100:>6.1f}% "
              f"{info['iv_put']*100:>6.1f}% {info['dte']:>4d} "
              f"{info['strike']:>7.1f}  {info['source']}")


if __name__ == "__main__":
    main()

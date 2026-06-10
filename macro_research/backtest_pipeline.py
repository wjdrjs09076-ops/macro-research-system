"""
backtest_pipeline.py — 과거 6개월 데이터로 ontology trigger 재실행 + 가상 발주 시뮬.

비평 라운드 4 의 메타 지적("페이퍼 첫 P&L 0건 = 시스템 가설 미검증") 을 페이퍼 사이클을
기다리지 않고 부분 우회. 6개월 historical 로 가상 trade_journal 채워서 attribution /
feedback / NORTA base rate vs 시그널 P&L 비교를 즉시 산출.

설계 (사용자 선택 2026-06-10)
─────────────────────────────
- 기간: END_DATE - 180일 ~ END_DATE (~126 거래일)
- 빈도: 일별 (장 마감 시점 트리거)
- 옵션 가격: Black-Scholes, σ = garch_vol.parquet 의 GARCH 일별 IV proxy
- 호가스프레드: IV 의 5% 슬리피지 (mid 가격에 비례 추가)
- 청산: TP +25% (straddle) / DTE≤10 / max-hold 21일 (directional) / TP+10%, SL-7% (directional)

알려진 한계 (look-ahead bias)
────────────────────────────
- tail_gpd / co_crash / SENSITIVE_TO 통계는 *모든* 과거 데이터로 한 번에 적합 → 백테스트
  의 임의 시점 t 에서 *t 이후* 정보를 포함할 수 있음. 시점-aware 재적합은 향후 과제.
- GARCH IV proxy 는 진짜 옵션 IV 가 아님 (라운드 4 비평 그대로 — 백테스트는 *상한 추정*).

산출
─────
- output/backtest_results.json — 요약 통계 + per-rule P&L
- output/backtest_journal.jsonl — 동일 trade_journal 형식 (attribution 재사용)
- 콘솔: NORTA base rate vs 실현 P&L 비교 표
"""
from __future__ import annotations

import datetime as dt
import json
import math
import sys
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUTPUT_DIR, SECTOR_ETFS
from ontology.schema import build_empty_graph, RegimeType
from ontology.inference import generate_signals
from ontology.trigger import (
    _aggregate_straddle, _aggregate_directional,
    LONG_STRADDLE_RULES, SIGNAL_DIRECTION,
)

# ── 파라미터 ──────────────────────────────────────────────────
LOOKBACK_DAYS  = 180
RISK_FREE      = 0.05
SLIPPAGE_PCT   = 0.05   # 호가스프레드 = IV의 5% → 옵션 mid 가격에 그만큼 추가/차감

# kinetic 와 동일한 청산 조건
STRADDLE_TP_PCT = 0.25
STRADDLE_DTE_EXIT = 10
DIR_TP_PCT = 0.10
DIR_SL_PCT = -0.07
DIR_MAX_HOLD = 21

# minimal 사이즈
BASE_WEIGHT = 0.005
DIR_BASE_WEIGHT = 0.005
MAX_SIGNALS = 2
CONF_REF = 0.65
CONF_SCALE_CAP = 2.0
DTE_TARGET = 30   # 가상 만기 = entry + 30일

OUTPUT_JSON = OUTPUT_DIR / "backtest_results.json"
OUTPUT_JOURNAL = OUTPUT_DIR / "backtest_journal.jsonl"


# ──────────────────────────────────────────────────────────────
# Black-Scholes
# ──────────────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             opt_type: str) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if opt_type == "call" else (K - S))
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt_type == "call":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


# ──────────────────────────────────────────────────────────────
# 시점-aware populate (look-ahead bias 부분 해소)
# ──────────────────────────────────────────────────────────────

def _build_graph_as_of(date: pd.Timestamp, returns: pd.DataFrame,
                       garch: pd.DataFrame, levels: pd.DataFrame) -> "nx.DiGraph":
    """as_of 시점 데이터로 populate. *시계열 의존* 캐시(GARCH/returns/VIX)는 cut.
    *통계 적합* 캐시(tail_gpd, sensitivity)는 그대로 (look-ahead bias 의 알려진 한계).
    """
    from ontology.schema import build_empty_graph
    from ontology.populate import (
        _populate_tail_risk, _populate_vrp, _populate_sensitivity,
        _populate_co_crash, _populate_causal_links,
    )
    G = build_empty_graph()
    # Tail/VRP/sensitivity 는 정적 캐시 그대로 — look-ahead 한계
    _populate_tail_risk(G)
    _populate_vrp(G)
    _populate_sensitivity(G)
    _populate_co_crash(G)
    _populate_causal_links(G)
    # 활성 레짐 = as_of 시점 VIX
    vix = float(levels.loc[:date, "VIX"].iloc[-1])
    regime = (RegimeType.LOW_VIX.value if vix < 15
              else RegimeType.HIGH_VIX.value if vix >= 25
              else RegimeType.MID_VIX.value)
    import networkx as nx
    for r in [RegimeType.LOW_VIX.value, RegimeType.MID_VIX.value,
              RegimeType.HIGH_VIX.value]:
        nx.set_node_attributes(G, {r: {"active": r == regime, "vix_latest": vix}})
    # Shapley 분산 분해 — 시점 cut 한 returns 사용
    try:
        from ontology.mvg_mc import shapley_decompose
        # 임시 monkey patch
        import ontology.mvg_mc as _mvg
        orig = _mvg._load_returns
        _mvg._load_returns = lambda: returns.loc[:date]
        try:
            d = shapley_decompose(n_samples=100)
        finally:
            _mvg._load_returns = orig
        for i, t in enumerate(d.order):
            attrs = {
                "var_decomposition": d.shock_contribution_to(t),
                "self_share":        d.self_share(t),
                "propagation_score": d.propagation_score(t),
                "cholesky_order":    i,
            }
            nx.set_node_attributes(G, {t: attrs})
    except Exception as exc:
        print(f"  [warn] Shapley skip @ {date.date()}: {exc}")
    # 진짜 IV (vrp_iv) — 백테스트엔 GARCH 기반 proxy 사용
    # ATM_IV ≈ GARCH 연환산 vol (as_of 시점)
    for sec in SECTOR_ETFS:
        if sec not in garch.columns:
            continue
        try:
            iv = float(garch.loc[:date, sec].iloc[-1])
            rv_window = returns.loc[:date, sec].tail(20)
            rv = float(rv_window.std() * np.sqrt(252)) if len(rv_window) >= 5 else float("nan")
            vrp = iv - rv if not (np.isnan(iv) or np.isnan(rv)) else float("nan")
            nx.set_node_attributes(G, {sec: {"iv_atm": round(iv, 4),
                                              "vrp_iv": round(vrp, 4)}})
        except Exception:
            pass
    return G, regime


# ──────────────────────────────────────────────────────────────
# Trade simulation
# ──────────────────────────────────────────────────────────────

@dataclass
class OpenStraddle:
    trade_id: str
    ticker: str
    rule: str
    all_rules: list
    signal_type: str
    confidence: float
    regime: str
    entry_date: pd.Timestamp
    expiry_date: pd.Timestamp
    spot_entry: float
    strike: float
    sigma_entry: float
    entry_cost: float       # 두 옵션 합계 (per share × 100 × qty), 슬리피지 포함
    qty: int


@dataclass
class OpenDirectional:
    trade_id: str
    ticker: str
    rule: str
    all_rules: list
    signal_type: str
    confidence: float
    regime: str
    entry_date: pd.Timestamp
    direction: int
    spot_entry: float
    qty: int
    entry_cost: float   # qty × spot × direction (롱은 양, 숏은 음)


def _straddle_mtm(s: OpenStraddle, today: pd.Timestamp, spot: float,
                  sigma: float) -> float:
    """현재 시점 스트래들 평가가치 (슬리피지 차감 청산 가정)."""
    T = max((s.expiry_date - today).days / 365.0, 1e-6)
    call = bs_price(spot, s.strike, T, RISK_FREE, sigma, "call")
    put  = bs_price(spot, s.strike, T, RISK_FREE, sigma, "put")
    mid = (call + put) * 100 * s.qty
    # 청산 시 슬리피지 (매도 → 호가 아래)
    return mid * (1 - SLIPPAGE_PCT)


def _directional_mtm(d: OpenDirectional, today: pd.Timestamp,
                     spot: float) -> tuple[float, float]:
    """(market_value, pnl_pct) 반환. 숏은 음의 mv."""
    notional = abs(d.qty) * spot
    if d.direction > 0:
        mv = notional
        pnl = (spot - d.spot_entry) / d.spot_entry
    else:
        mv = -notional
        pnl = (d.spot_entry - spot) / d.spot_entry
    return mv, pnl


# ──────────────────────────────────────────────────────────────
# Main backtest loop
# ──────────────────────────────────────────────────────────────

def run_backtest() -> dict:
    print(f"[backtest] 로딩 캐시 ...")
    returns = pd.read_parquet(OUTPUT_DIR / "sector_returns.parquet")
    garch   = pd.read_parquet(OUTPUT_DIR / "garch_vol.parquet")
    levels  = pd.read_parquet(OUTPUT_DIR / "macro_levels.parquet")
    prices  = pd.read_parquet(OUTPUT_DIR / "sector_prices.parquet")

    # 공통 날짜
    dates = returns.index.intersection(garch.index).intersection(levels.index)
    end = dates[-1]
    start = end - pd.Timedelta(days=LOOKBACK_DAYS)
    sim_dates = dates[(dates >= start) & (dates <= end)]
    print(f"[backtest] 시뮬 기간 {sim_dates[0].date()} ~ {sim_dates[-1].date()} "
          f"({len(sim_dates)}일)")

    open_straddles: list[OpenStraddle] = []
    open_directional: list[OpenDirectional] = []
    journal_events: list[dict] = []
    equity = 100_000.0   # 가상 초기 자본

    # ────────────────────────────────────────────────────
    for di, today in enumerate(sim_dates):
        # 1. 보유 포지션 mark-to-market + 청산 평가
        kept_str: list[OpenStraddle] = []
        for s in open_straddles:
            tkr = s.ticker
            if tkr not in prices.columns:
                kept_str.append(s); continue
            spot = float(prices.loc[:today, tkr].iloc[-1])
            sigma = float(garch.loc[:today, tkr].iloc[-1]) if tkr in garch.columns else s.sigma_entry
            mv = _straddle_mtm(s, today, spot, sigma)
            pnl_pct = (mv - s.entry_cost) / s.entry_cost if s.entry_cost else 0
            dte = (s.expiry_date - today).days
            reason = None
            if pnl_pct >= STRADDLE_TP_PCT:
                reason = "익절"
            elif dte <= STRADDLE_DTE_EXIT:
                reason = "만기임박"
            if reason:
                journal_events.append({
                    "event": "exit", "trade_id": s.trade_id, "ts": today.isoformat(),
                    "strategy": "straddle", "ticker": tkr, "exit_reason": reason,
                    "entry_cost": round(s.entry_cost, 2), "exit_value": round(mv, 2),
                    "pnl": round(mv - s.entry_cost, 2), "pnl_pct": round(pnl_pct, 4),
                    "holding_days": (today - s.entry_date).days, "min_dte": dte,
                })
                equity += (mv - s.entry_cost)
            else:
                kept_str.append(s)
        open_straddles = kept_str

        kept_dir: list[OpenDirectional] = []
        for d in open_directional:
            tkr = d.ticker
            if tkr not in prices.columns:
                kept_dir.append(d); continue
            spot = float(prices.loc[:today, tkr].iloc[-1])
            mv, pnl_pct = _directional_mtm(d, today, spot)
            held_days = (today - d.entry_date).days
            reason = None
            if pnl_pct >= DIR_TP_PCT:
                reason = "익절"
            elif pnl_pct <= DIR_SL_PCT:
                reason = "손절"
            elif held_days >= DIR_MAX_HOLD:
                reason = "보유만료"
            if reason:
                exit_value = abs(d.entry_cost) * (1 + pnl_pct)
                journal_events.append({
                    "event": "exit", "trade_id": d.trade_id, "ts": today.isoformat(),
                    "strategy": "directional", "ticker": tkr, "exit_reason": reason,
                    "entry_cost": round(abs(d.entry_cost), 2),
                    "exit_value": round(exit_value, 2),
                    "pnl": round(exit_value - abs(d.entry_cost), 2),
                    "pnl_pct": round(pnl_pct, 4), "holding_days": held_days,
                })
                equity += (exit_value - abs(d.entry_cost))
            else:
                kept_dir.append(d)
        open_directional = kept_dir

        # 2. 새 진입: ontology 트리거 (as_of today)
        try:
            G, regime = _build_graph_as_of(today, returns, garch, levels)
            signals = generate_signals(G, regime, run_all_regimes=False)
        except Exception as exc:
            print(f"  [warn] graph build failed @ {today.date()}: {exc}")
            continue
        if not signals:
            continue
        straddle = _aggregate_straddle(signals)
        directional = _aggregate_directional(signals)

        held_opt = {s.ticker for s in open_straddles}
        held_eq = {d.ticker for d in open_directional}
        straddle = [c for c in straddle if c["ticker"] not in held_opt][:MAX_SIGNALS]
        directional = [c for c in directional if c["ticker"] not in held_eq][:MAX_SIGNALS]

        expiry = today + pd.Timedelta(days=DTE_TARGET)

        for cand in straddle:
            tkr = cand["ticker"]
            if tkr not in prices.columns or tkr not in garch.columns: continue
            spot = float(prices.loc[:today, tkr].iloc[-1])
            sigma = float(garch.loc[:today, tkr].iloc[-1])
            if not (spot > 0 and sigma > 0): continue
            strike = round(spot)
            T = DTE_TARGET / 365.0
            call = bs_price(spot, strike, T, RISK_FREE, sigma, "call")
            put  = bs_price(spot, strike, T, RISK_FREE, sigma, "put")
            unit = (call + put) * 100 * (1 + SLIPPAGE_PCT)   # ask 슬리피지
            qty = max(1, int(equity * BASE_WEIGHT / unit))
            entry_cost = unit * qty
            tid = f"{tkr}-{today.date()}-{uuid.uuid4().hex[:6]}"
            open_straddles.append(OpenStraddle(
                trade_id=tid, ticker=tkr,
                rule=cand["rule"], all_rules=cand.get("all_rules", []),
                signal_type=cand["signal_type"],
                confidence=cand["confidence"], regime=cand["regime"],
                entry_date=today, expiry_date=expiry,
                spot_entry=spot, strike=strike, sigma_entry=sigma,
                entry_cost=entry_cost, qty=qty,
            ))
            journal_events.append({
                "event": "entry", "trade_id": tid, "ts": today.isoformat(),
                "strategy": "straddle", "ticker": tkr,
                "rule": cand["rule"], "all_rules": cand.get("all_rules", []),
                "signal_type": cand["signal_type"], "confidence": cand["confidence"],
                "regime": cand["regime"], "qty": qty, "entry_cost": round(entry_cost, 2),
                "spot": round(spot, 2),
            })
            equity -= entry_cost

        for cand in directional:
            tkr = cand["ticker"]
            if tkr not in prices.columns: continue
            spot = float(prices.loc[:today, tkr].iloc[-1])
            if spot <= 0: continue
            qty_val = max(1, int(equity * DIR_BASE_WEIGHT / spot))
            qty = qty_val * cand["direction"]
            entry_cost = qty_val * spot   # 양수 cost
            tid = f"{tkr}-{today.date()}-{uuid.uuid4().hex[:6]}"
            open_directional.append(OpenDirectional(
                trade_id=tid, ticker=tkr,
                rule=cand["rule"], all_rules=cand.get("all_rules", []),
                signal_type=cand["signal_type"],
                confidence=cand["confidence"], regime=cand["regime"],
                entry_date=today, direction=cand["direction"],
                spot_entry=spot, qty=qty, entry_cost=entry_cost,
            ))
            journal_events.append({
                "event": "entry", "trade_id": tid, "ts": today.isoformat(),
                "strategy": "directional", "ticker": tkr,
                "rule": cand["rule"], "all_rules": cand.get("all_rules", []),
                "signal_type": cand["signal_type"], "confidence": cand["confidence"],
                "regime": cand["regime"], "qty": qty,
                "entry_cost": round(entry_cost, 2), "direction": cand["direction"],
                "spot": round(spot, 2), "entry_price": round(spot, 2),
            })

        if di % 20 == 0:
            print(f"  [{today.date()}] open straddle={len(open_straddles)} "
                  f"dir={len(open_directional)} equity=${equity:,.0f} "
                  f"events={len(journal_events)}")

    # 저장
    OUTPUT_JOURNAL.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in journal_events),
        encoding="utf-8",
    )
    return _summarize(journal_events, sim_dates)


def _summarize(events: list[dict], sim_dates: pd.DatetimeIndex) -> dict:
    by_id: dict[str, dict] = {}
    closed: list[dict] = []
    for e in events:
        tid = e["trade_id"]
        if e["event"] == "entry":
            by_id[tid] = e
        elif e["event"] == "exit" and tid in by_id:
            ent = by_id[tid]
            closed.append({**ent, **{k: v for k, v in e.items() if k != "ts"}, "exit_ts": e["ts"]})

    n_closed = len(closed)
    by_strategy = {}
    by_rule_strategy = {}
    for c in closed:
        key_s = c.get("strategy", "?")
        key_rs = f"{c.get('rule')}|{key_s}"
        for d, k in [(by_strategy, key_s), (by_rule_strategy, key_rs)]:
            d.setdefault(k, []).append(c)

    def _agg(rows):
        pnls = [float(r.get("pnl") or 0) for r in rows]
        pcts = [float(r.get("pnl_pct") or 0) for r in rows]
        wins = sum(1 for p in pnls if p > 0)
        n = len(rows)
        return {
            "trades": n, "wins": wins,
            "win_rate":    round(wins / n, 4) if n else 0,
            "avg_pnl_pct": round(sum(pcts) / n, 4) if n else 0,
            "total_pnl":   round(sum(pnls), 2),
        }

    summary = {
        "period": {"start": str(sim_dates[0].date()), "end": str(sim_dates[-1].date()),
                   "n_days": len(sim_dates)},
        "n_closed": n_closed,
        "by_strategy":      {k: _agg(v) for k, v in by_strategy.items()},
        "by_rule_strategy": {k: _agg(v) for k, v in by_rule_strategy.items()},
    }
    return summary


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print(f"{'='*64}")
    print(f"  백테스트 — 과거 {LOOKBACK_DAYS}일, 일별 trigger, BS+GARCH IV")
    print(f"{'='*64}")
    summary = run_backtest()

    OUTPUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"\n=== 요약 ===")
    print(f"기간: {summary['period']['start']} ~ {summary['period']['end']} "
          f"({summary['period']['n_days']}일)")
    print(f"완결 거래: {summary['n_closed']}")
    print(f"\n전략별:")
    for k, v in summary["by_strategy"].items():
        print(f"  {k:12s} n={v['trades']:>3} win={v['win_rate']*100:>4.0f}%  "
              f"avg={v['avg_pnl_pct']*100:>+5.1f}%  total=${v['total_pnl']:>+8,.0f}")
    print(f"\n(rule×strategy) 별:")
    for k, v in sorted(summary["by_rule_strategy"].items(),
                       key=lambda kv: -kv[1]["total_pnl"]):
        print(f"  {k:42s} n={v['trades']:>3} win={v['win_rate']*100:>4.0f}%  "
              f"avg={v['avg_pnl_pct']*100:>+5.1f}%  total=${v['total_pnl']:>+8,.0f}")

    print(f"\n→ {OUTPUT_JSON}")
    print(f"→ {OUTPUT_JOURNAL}")


if __name__ == "__main__":
    main()

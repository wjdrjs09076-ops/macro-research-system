"""
trade_journal.py - 키네틱 레이어의 거래 저널 (append-only JSONL).

모든 진입/청산을 시그널 맥락과 함께 기록해 시그널 → 결과를 연결한다.
귀인(attribution.py)·피드백(feedback.py)의 데이터 소스.

전략(strategy) 차원: 같은 티커에 straddle(옵션)과 directional(현물)을 동시에
보유할 수 있으므로, 미청산 매칭 키는 (ticker, strategy) 다.

레코드 구조 (output/trade_journal.jsonl, 한 줄당 1 이벤트)
─────────────────────────────────────────────────────
ENTRY:
  event=entry, trade_id, ts, strategy, ticker, rule, all_rules[], signal_type,
  confidence, regime, conf_multiplier, direction, spot, entry_price,
  strike, expiry, call_sym, put_sym, qty, budget, entry_cost, reasoning[]
EXIT:
  event=exit, trade_id, ts, strategy, ticker, exit_reason, entry_cost,
  exit_value, pnl, pnl_pct, holding_days, min_dte

trade_id가 진입↔청산을 잇는다. 청산 시 (underlying, strategy)의 미청산 진입을 찾아 매칭.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUTPUT_DIR

JOURNAL = OUTPUT_DIR / "trade_journal.jsonl"


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _append(rec: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_records() -> list[dict]:
    if not JOURNAL.exists():
        return []
    out: list[dict] = []
    for line in JOURNAL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def new_trade_id(ticker: str) -> str:
    return f"{ticker}-{dt.date.today().isoformat()}-{uuid.uuid4().hex[:6]}"


def open_trades() -> dict[tuple[str, str], dict]:
    """미청산 진입을 (ticker, strategy) -> entry record 로 반환.

    저널을 순서대로 재생: entry는 open에 추가, exit는 매칭 trade_id 제거.
    같은 (ticker, strategy)에 진입이 여럿이면 가장 최근 미청산 건을 남긴다.
    """
    open_by_id: dict[str, dict] = {}
    for rec in load_records():
        tid = rec.get("trade_id")
        if rec.get("event") == "entry" and tid:
            open_by_id[tid] = rec
        elif rec.get("event") == "exit" and tid:
            open_by_id.pop(tid, None)

    by_key: dict[tuple[str, str], dict] = {}
    for rec in open_by_id.values():
        key = (rec["ticker"], rec.get("strategy", "straddle"))
        by_key[key] = rec  # 후행 진입이 선행을 덮음 = 최신 유지
    return by_key


def open_entries_grouped() -> dict[tuple[str, str], list[dict]]:
    """미청산 진입을 (ticker, strategy) -> [entry, ...] (진입시각 오름차순) 로 반환.

    open_trades() 는 같은 키의 중복 미청산을 '최신만' 남겨 덮어쓴다(상태 조회용).
    여기서는 *모든* 미청산 진입을 보존한다 — churn(같은 종목 빠른 재진입)으로 저널에
    유령(고아 진입)이 누적됐는지 라이브 보유수와 대조하려면 키로 접지 않은 전체가 필요.
    """
    open_by_id: dict[str, dict] = {}
    for rec in load_records():
        tid = rec.get("trade_id")
        if rec.get("event") == "entry" and tid:
            open_by_id[tid] = rec
        elif rec.get("event") == "exit" and tid:
            open_by_id.pop(tid, None)
    grouped: dict[tuple[str, str], list[dict]] = {}
    for rec in open_by_id.values():
        key = (rec["ticker"], rec.get("strategy", "straddle"))
        grouped.setdefault(key, []).append(rec)
    for recs in grouped.values():
        recs.sort(key=lambda r: r.get("ts", ""))
    return grouped


# 유령 정리(저널-라이브 정합) 청산 사유. 귀인/전향검증에서 제외(테제 결과 아님).
PHANTOM_EXIT_REASON = "유령정리(저널-라이브 정합)"


def reconcile_phantoms(live_counts: dict[tuple[str, str], int],
                       positions_ok: bool) -> list[dict]:
    """저널 미청산 ↔ 라이브(Alpaca) 보유수 대조 → 초과(유령) 진입을 P&L-중립 청산.

    라이브가 정답. 같은 (ticker, strategy)에 저널 미청산이 라이브 보유수보다 많으면,
    초과분(가장 오래된 것부터)은 churn 중 Alpaca 청산이 저널 exit 를 못 남긴 유령이다
    → pnl=0, reason=PHANTOM_EXIT_REASON 으로 닫아 저널을 라이브에 맞춘다.
    실손익은 매칭됐던 (잘못된) exit 에 이미 기록됐으므로 여기선 0 (이중계상 방지).

    ★ 안전장치: positions_ok=False(=Alpaca positions API 실패. get_positions 가 []로
    위장하는 케이스)면 *아무것도 닫지 않는다*. API 실패를 '전부 청산됨'으로 오인해 정상
    미청산을 몰살하는 것을 막는다 — 이 수정의 핵심 footgun.

    Returns 정리한 유령 entry 레코드 리스트.
    """
    if not positions_ok:
        return []
    grouped = open_entries_grouped()
    cleaned: list[dict] = []
    for key, recs in grouped.items():
        live_n = max(0, int(live_counts.get(key, 0)))
        excess = len(recs) - live_n
        if excess <= 0:
            continue
        # 최신 live_n 건 = 실제 보유로 간주, 나머지(오래된 것부터) = 유령.
        for rec in recs[:excess]:
            entry_cost = round(float(rec.get("entry_cost") or 0), 2)
            _append({
                "event":        "exit",
                "trade_id":     rec.get("trade_id"),
                "ts":           _now(),
                "strategy":     rec.get("strategy", "straddle"),
                "ticker":       rec.get("ticker"),
                "exit_reason":  PHANTOM_EXIT_REASON,
                "entry_cost":   entry_cost,
                "exit_value":   entry_cost,   # pnl=0
                "pnl":          0.0,
                "pnl_pct":      0.0,
                "holding_days": None,
                "min_dte":      0,
            })
            cleaned.append(rec)
    return cleaned


def log_entry(
    *,
    strategy: str,
    ticker: str,
    rule: str,
    all_rules: list[str],
    signal_type: str,
    confidence: float,
    regime: str,
    conf_multiplier: float,
    qty: int,
    budget: float,
    entry_cost: float,
    reasoning: list[str],
    direction: int = 0,
    spot: float = 0.0,
    entry_price: float = 0.0,
    strike: float | None = None,
    expiry: str | None = None,
    call_sym: str | None = None,
    put_sym: str | None = None,
    hedge_symbol: str | None = None,
    hedge_qty: float | None = None,
    hedge_price: float | None = None,
    hedge_beta: float | None = None,
    hedge_resid_vol: float | None = None,
) -> str:
    """진입 기록. 생성된 trade_id 반환.

    strategy='straddle' 이면 strike/expiry/call_sym/put_sym 사용,
    'directional' 이면 direction/entry_price 사용.
    hedge_* (2026-06-12): 베타헤지 페어 — directional 진입에 붙는 SPY 반대 레그.
    """
    trade_id = new_trade_id(ticker)
    rec = {
        "event":           "entry",
        "trade_id":        trade_id,
        "ts":              _now(),
        "strategy":        strategy,
        "ticker":          ticker,
        "rule":            rule,
        "all_rules":       all_rules,
        "signal_type":     signal_type,
        "confidence":      round(float(confidence), 4),
        "regime":          regime,
        "conf_multiplier": round(float(conf_multiplier), 4),
        "direction":       int(direction),
        "spot":            round(float(spot), 2),
        "entry_price":     round(float(entry_price), 2),
        "strike":          round(float(strike), 2) if strike is not None else None,
        "expiry":          expiry,
        "call_sym":        call_sym,
        "put_sym":         put_sym,
        "qty":             int(qty),
        "budget":          round(float(budget), 2),
        "entry_cost":      round(float(entry_cost), 2),
        "reasoning":       reasoning,
    }
    if hedge_qty:
        rec.update({
            "hedge_symbol":    hedge_symbol or "SPY",
            "hedge_qty":       round(float(hedge_qty), 3),   # 소수주 (int 캐스트 시 0.3→0 = 페어 무력화·churn)
            "hedge_price":     round(float(hedge_price or 0), 2),
            "hedge_beta":      round(float(hedge_beta or 0), 4),
            "hedge_resid_vol": round(float(hedge_resid_vol or 0), 6),
        })
    _append(rec)
    return trade_id


def log_exit(
    *,
    strategy: str,
    ticker: str,
    exit_reason: str,
    exit_value: float,
    pnl_pct: float,
    min_dte: int = 0,
    entry_cost_fallback: float | None = None,
) -> str | None:
    """청산 기록. (underlying, strategy)의 미청산 진입을 찾아 trade_id로 연결.

    Returns 연결된 trade_id (없으면 None).
    """
    entry = open_trades().get((ticker, strategy))
    trade_id = entry.get("trade_id") if entry else None
    entry_cost = (entry.get("entry_cost") if entry else None)
    if entry_cost is None:
        entry_cost = entry_cost_fallback if entry_cost_fallback is not None else 0.0

    holding_days = None
    if entry and entry.get("ts"):
        try:
            t0 = dt.datetime.fromisoformat(entry["ts"]).date()
            holding_days = (dt.date.today() - t0).days
        except ValueError:
            holding_days = None

    pnl = float(exit_value) - float(entry_cost)
    _append({
        "event":        "exit",
        "trade_id":     trade_id,
        "ts":           _now(),
        "strategy":     strategy,
        "ticker":       ticker,
        "exit_reason":  exit_reason,
        "entry_cost":   round(float(entry_cost), 2),
        "exit_value":   round(float(exit_value), 2),
        "pnl":          round(pnl, 2),
        "pnl_pct":      round(float(pnl_pct), 4),
        "holding_days": holding_days,
        "min_dte":      int(min_dte),
    })
    return trade_id


def closed_trades() -> list[dict]:
    """진입↔청산을 trade_id로 조인한 완결 거래 리스트 (귀인용)."""
    entries: dict[str, dict] = {}
    out: list[dict] = []
    for rec in load_records():
        tid = rec.get("trade_id")
        if not tid:
            continue
        if rec.get("event") == "entry":
            entries[tid] = rec
        elif rec.get("event") == "exit" and tid in entries:
            if rec.get("exit_reason") == PHANTOM_EXIT_REASON:
                entries.pop(tid, None)   # 유령 정리분: 회계 정합용 청산 — 귀인/검증 모두 제외
                continue
            e = entries[tid]
            out.append({
                "trade_id":     tid,
                "entry_ts":     e.get("ts"),
                "exit_ts":      rec.get("ts"),
                "strategy":     e.get("strategy", "straddle"),
                "ticker":       e["ticker"],
                "rule":         e.get("rule"),
                "all_rules":    e.get("all_rules", []),
                "signal_type":  e.get("signal_type"),
                "direction":    e.get("direction", 0),
                "confidence":   e.get("confidence"),
                "regime":       e.get("regime"),
                "entry_cost":   rec.get("entry_cost", e.get("entry_cost")),
                "hedge_qty":    e.get("hedge_qty"),
                "hedge_beta":   e.get("hedge_beta"),
                "exit_value":   rec.get("exit_value"),
                "pnl":          rec.get("pnl"),
                "pnl_pct":      rec.get("pnl_pct"),
                "exit_reason":  rec.get("exit_reason"),
                "holding_days": rec.get("holding_days"),
            })
    return out

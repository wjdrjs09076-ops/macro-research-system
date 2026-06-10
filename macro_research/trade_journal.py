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
) -> str:
    """진입 기록. 생성된 trade_id 반환.

    strategy='straddle' 이면 strike/expiry/call_sym/put_sym 사용,
    'directional' 이면 direction/entry_price 사용.
    """
    trade_id = new_trade_id(ticker)
    _append({
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
    })
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
            e = entries[tid]
            out.append({
                "trade_id":     tid,
                "strategy":     e.get("strategy", "straddle"),
                "ticker":       e["ticker"],
                "rule":         e.get("rule"),
                "all_rules":    e.get("all_rules", []),
                "signal_type":  e.get("signal_type"),
                "direction":    e.get("direction", 0),
                "confidence":   e.get("confidence"),
                "regime":       e.get("regime"),
                "entry_cost":   rec.get("entry_cost", e.get("entry_cost")),
                "exit_value":   rec.get("exit_value"),
                "pnl":          rec.get("pnl"),
                "pnl_pct":      rec.get("pnl_pct"),
                "exit_reason":  rec.get("exit_reason"),
                "holding_days": rec.get("holding_days"),
            })
    return out

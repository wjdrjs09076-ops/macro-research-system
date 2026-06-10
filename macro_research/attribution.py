"""
attribution.py - 거래 저널을 룰/레짐 단위로 귀인 분석.

trade_journal.closed_trades() (진입↔청산 조인)를 읽어
  - 룰별: 거래수, 승률, 평균 P&L%, 총 P&L, 평균 보유일
  - 레짐별: 동일 지표
를 계산하고 output/attribution.json 에 저장 + 표 출력.

귀인은 primary rule(진입 시 대표 룰) 기준. 같은 진입에 여러 룰이 함께
발화한 경우 all_rules 기준 분해는 향후 과제(현재는 대표 룰만 집계).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUTPUT_DIR
from trade_journal import closed_trades

ATTRIBUTION_JSON = OUTPUT_DIR / "attribution.json"


def _agg(trades: list[dict], key, label: str | None = None) -> dict[str, dict]:
    """key 가 문자열이면 해당 필드로, 호출가능이면 t->버킷키 로 집계."""
    keyfn = key if callable(key) else (lambda t, k=key: t.get(k) or "?")
    buckets: dict[str, list[dict]] = {}
    for t in trades:
        k = keyfn(t)
        buckets.setdefault(k, []).append(t)

    out: dict[str, dict] = {}
    for k, ts in buckets.items():
        pnls = [float(t.get("pnl") or 0) for t in ts]
        pnl_pcts = [float(t.get("pnl_pct") or 0) for t in ts]
        holds = [t["holding_days"] for t in ts if t.get("holding_days") is not None]
        wins = sum(1 for p in pnls if p > 0)
        n = len(ts)
        out[k] = {
            "trades":        n,
            "wins":          wins,
            "win_rate":      round(wins / n, 4) if n else 0.0,
            "avg_pnl_pct":   round(sum(pnl_pcts) / n, 4) if n else 0.0,
            "total_pnl":     round(sum(pnls), 2),
            "avg_hold_days": round(sum(holds) / len(holds), 1) if holds else None,
        }
    return out


def _rule_strategy_key(t: dict) -> str:
    """feedback/사이징과 동일한 합성키 'rule|strategy'."""
    return f"{t.get('rule') or '?'}|{t.get('strategy') or 'straddle'}"


def build_attribution(write: bool = True) -> dict:
    trades = closed_trades()
    report = {
        "n_closed":         len(trades),
        "by_strategy":      _agg(trades, "strategy"),
        "by_rule_strategy": _agg(trades, _rule_strategy_key),
        "by_rule":          _agg(trades, "rule"),
        "by_regime":        _agg(trades, "regime"),
        "by_signal":        _agg(trades, "signal_type"),
    }
    if write:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ATTRIBUTION_JSON.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _print_table(title: str, stats: dict[str, dict]) -> None:
    print(f"\n  {title}")
    print(f"  {'key':32s} {'n':>4} {'win%':>6} {'avgP&L%':>9} {'totP&L$':>10} {'hold':>6}")
    print(f"  {'-'*70}")
    for k, s in sorted(stats.items(), key=lambda kv: -kv[1]["total_pnl"]):
        hold = f"{s['avg_hold_days']:.1f}" if s["avg_hold_days"] is not None else "-"
        print(f"  {k:32s} {s['trades']:>4} {s['win_rate']*100:>5.0f}% "
              f"{s['avg_pnl_pct']*100:>8.1f}% {s['total_pnl']:>10,.0f} {hold:>6}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    report = build_attribution()
    print(f"\n{'='*64}")
    print(f"  귀인 분석 (ATTRIBUTION)  |  완결 거래 {report['n_closed']}건")
    print(f"{'='*64}")

    if report["n_closed"] == 0:
        print("\n  아직 완결된 거래 없음. 진입+청산이 한 쌍 이상 쌓이면 집계됩니다.")
        print(f"\n-> {ATTRIBUTION_JSON}")
        return

    _print_table("전략별 (strategy)", report["by_strategy"])
    _print_table("룰×전략별 (rule|strategy)", report["by_rule_strategy"])
    _print_table("룰별 (rule)", report["by_rule"])
    _print_table("레짐별 (regime)", report["by_regime"])
    _print_table("시그널타입별 (signal_type)", report["by_signal"])
    print(f"\n-> {ATTRIBUTION_JSON}")


if __name__ == "__main__":
    main()

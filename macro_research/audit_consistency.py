# -*- coding: utf-8 -*-
"""audit_consistency.py — 단일 진실원 대조 스크립트 (감사 P0-1 완료 기준).

검사 항목:
  [1] Alpaca 보유 포지션 ↔ rule_sector_state.valid_pairs 1:1 (P0-2 완료 기준)
  [2] registry rate_verdicts ↔ ontology_signals.sensitivity_audit verdict 정합
  [3] 가드 단위 테스트 (예산 가드 / rate 패밀리 캡) — 네트워크 모킹 (P2-1)
  [4] 숏볼 휴면 트랙 — CI 게이트 통과에 필요한 표본 N 근사 (P2-5)

사용: python audit_consistency.py
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config import OUTPUT_DIR

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results: list[tuple[str, str, str]] = []


def check(tag: str, ok: bool, detail: str, warn: bool = False):
    status = PASS if ok else (WARN if warn else FAIL)
    results.append((tag, status, detail))
    print(f"[{status}] {tag}: {detail}")


# ── [1] 포지션 ↔ valid_pairs ───────────────────────────────────────
def check_positions():
    import kinetic_executor as K
    import trade_journal

    state = json.loads((OUTPUT_DIR / "rule_sector_state.json").read_text(encoding="utf-8"))
    valid = set(state.get("valid_pairs", []))

    open_tr = trade_journal.open_trades()
    positions = [p for p in K.get_positions() if p.get("asset_class") == "us_equity"
                 and p["symbol"] != K.HEDGE_SYMBOL]
    opts = [p for p in K.get_positions() if p.get("asset_class") == "us_option"]

    bad = []
    for p in positions:
        tkr = p["symbol"]
        rec = open_tr.get((tkr, "directional"))
        rule = rec.get("rule") if rec else None
        if rule and rule != "vol_monitor_z" and f"{rule}|{tkr}" not in valid:
            bad.append(f"{rule}|{tkr}")
    check("P0-2 포지션↔ALIVE 1:1",
          not bad,
          f"현물 {len(positions)}건 + 옵션레그 {len(opts)}건, 사망 시그널 보유 {bad or '0건'}")


# ── [2] registry ↔ 감사표 verdict 정합 ─────────────────────────────
def check_registry_vs_audit():
    state = json.loads((OUTPUT_DIR / "rule_sector_state.json").read_text(encoding="utf-8"))
    sig = json.loads((OUTPUT_DIR / "ontology_signals.json").read_text(encoding="utf-8"))
    verdicts = state.get("rate_verdicts", [])
    if not verdicts:
        check("P0-1 registry 스키마", False, "rate_verdicts 부재 — trigger 재실행 필요")
        return
    audit = {(r["sector"], r["macro"]): r for r in sig.get("sensitivity_audit", [])}

    mismatch = []
    for v in verdicts:
        if v["rule"] != "rate_victim":   # 감사표 verdict 와 직접 비교 가능한 축
            continue
        a = audit.get((v["sector"], v["macro"]))
        # ALIVE ⇒ 감사표에서 confirmed/emerged 여야 함
        if v["verdict"] == "ALIVE":
            if not a or a["verdict"] not in ("confirmed", "emerged"):
                mismatch.append(f"{v['sector']}×{v['macro']} ALIVE vs audit={a['verdict'] if a else '없음'}")
    xlu = next((v for v in verdicts
                if v["sector"] == "XLU" and v["rule"] == "rate_victim"), None)
    check("P0-1 registry↔감사표 정합", not mismatch, f"불일치 {mismatch or '0건'}")
    check("P0-1 XLU verdict 명시",
          xlu is not None,
          f"XLU rate_victim = {xlu['verdict']} (t={xlu['evidence']['t']}, q={xlu['evidence']['q_fdr']})"
          if xlu else "registry 에 XLU 행 없음")


def check_valid_pairs_vs_verdicts():
    """SSOT 자기정합: valid_pairs(인퍼런스 발화) 와 rate_verdicts(FDR) 가 rate_*
    전 룰에서 일치해야 한다. rate_beneficiary US2Y 뒷문(raw-t 폴백)이 valid_pairs
    에 killed 페어를 넣던 결함을 잡는다 (P0 옵션1). (rule,sector) 는 어느 macro든
    하나라도 ALIVE 면 alive."""
    state = json.loads((OUTPUT_DIR / "rule_sector_state.json").read_text(encoding="utf-8"))
    sig   = json.loads((OUTPUT_DIR / "ontology_signals.json").read_text(encoding="utf-8"))
    valid = set(state.get("valid_pairs", []))
    verdicts = state.get("rate_verdicts", [])

    alive_any: dict[tuple[str, str], bool] = {}
    for v in verdicts:
        key = (v["rule"], v["sector"])
        alive_any[key] = alive_any.get(key, False) or (v["verdict"] == "ALIVE")

    bad = []
    for (rule, sec), alive in alive_any.items():
        in_valid = f"{rule}|{sec}" in valid
        if in_valid != alive:
            bad.append(f"{rule}|{sec}: valid_pairs={in_valid} vs verdict_ALIVE={alive}")

    # 라이브 directional 도 교차검증 — killed rate_* 가 매매 후보에 없어야 함
    live_bad = []
    for c in sig.get("directional_candidates", []):
        rule, sec = c.get("rule", ""), c.get("ticker", "")
        if rule.startswith("rate_") and alive_any.get((rule, sec), False) is False:
            live_bad.append(f"{rule}|{sec}(conf {c.get('confidence')})")

    check("P0 valid_pairs↔verdicts 정합", not bad, f"불일치 {bad or '0건'}")
    check("P0 라이브 directional = ALIVE only", not live_bad,
          f"killed 시그널 라이브 {live_bad or '0건'}")


# ── [3] 가드 단위 테스트 (네트워크 모킹) ───────────────────────────
def check_guards():
    import kinetic_executor as K
    import trade_journal

    spots = {"XLP": 85.0, "SPY": 600.0, "BIG": 5000.0}
    K.get_spot = lambda s: spots.get(s, 50.0)

    hs = {"XLP": {"beta": 0.5, "resid_vol": 0.005}}
    cand = {"ticker": "XLP", "confidence": 0.65, "rule": "rate_victim",
            "direction": -1, "signal_type": "UNDERWEIGHT", "regime": "mid_vix"}

    # 예산 가드: budget = equity×5% — spot 85 > 25×2 → 스킵 / 85 ≤ 100×2 → 통과
    p_ok = K.build_directional_plan(dict(cand), equity=2000, multipliers={}, hedge_stats=hs)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        p_skip = K.build_directional_plan(dict(cand), equity=500, multipliers={}, hedge_stats=hs)
    check("P2-1 예산 가드", p_ok is not None and p_skip is None,
          f"equity 2000→qty={getattr(p_ok,'qty',None)} / equity 500→스킵({'사이징 역전' in buf.getvalue()})")

    # 패밀리 캡: 기존 rate 노출 $14,500 (저널 모킹) + 신규 $4,xxx > 15% of 100k → 스킵
    K.held_equity_underlyings = lambda: set()
    trade_journal.open_trades = lambda: {
        ("XLB", "directional"): {"rule": "rate_victim", "entry_cost": 14500, "ts": "2026-06-12T10:00:00"},
    }
    K.trade_journal.open_trades = trade_journal.open_trades
    K.get_market_open = lambda: False
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        K._run_directional_entry([dict(cand)], equity=100_000, multipliers={},
                                 live=False, market_open=False)
    out = buf2.getvalue()
    check("P2-1 rate 패밀리 캡", "패밀리 캡" in out,
          "기존 $14,500 + 신규 > $15,000 한도 → 진입 스킵 확인" if "패밀리 캡" in out
          else "캡 미발동 — 로직 확인 필요")


# ── [4] 숏볼 휴면 — 필요 표본 N 근사 (감사 P2-5) ──────────────────
def dormancy_n():
    # λ_L: SE ≈ √(λ(1−λ)/(n·u)), u=0.05. 게이트: 95% 상한 < 0.20.
    # 진짜 λ=0.10 가정 → 마진 0.10 → n ≥ λ(1−λ)/u × (1.645/0.10)²
    lam, u = 0.10, 0.05
    n_lambda = lam * (1 - lam) / u * (1.645 / (0.20 - lam)) ** 2
    # ξ: SE ≈ (1+ξ)/√n_exc, n_exc = 0.1n. 게이트: 상한 < 0.10. 진짜 ξ=0 가정.
    xi = 0.0
    n_xi = ((1.645 * (1 + xi)) / (0.10 - xi)) ** 2 / 0.10
    check("P2-5 휴면 재개 표본", True,
          f"λ 게이트 통과에 ~{n_lambda:,.0f}영업일(≈{n_lambda/252:.1f}년), "
          f"ξ 게이트에 ~{n_xi:,.0f}영업일(≈{n_xi/252:.1f}년) — "
          f"현 표본(~370일)에서 사실상 영구 폐쇄 = 의도된 휴면이 정확한 라벨",
          warn=False)


if __name__ == "__main__":
    print("=" * 64)
    print("  감사 정합성 대조 (P0-1/P0-2/P2-1/P2-5 완료 기준)")
    print("=" * 64)
    check_registry_vs_audit()
    check_valid_pairs_vs_verdicts()
    check_positions()
    check_guards()
    dormancy_n()
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    print("-" * 64)
    print(f"  결과: {len(results)}건 중 FAIL {n_fail}건")
    sys.exit(1 if n_fail else 0)

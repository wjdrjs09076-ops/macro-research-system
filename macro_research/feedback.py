"""
feedback.py - 귀인 결과 → (룰×전략)별 *사후 사이징 보정* + 나쁜 룰 stop-rule.

⚠ "학습 루프" 라기보다 "사후 사이징 보정". 핵심 룰들(natural_hedge, crash_vulnerable 등
high_vix 발화 룰)은 표본 누적 자체가 수년 단위로 걸리므로 사실상 *열린 루프*. 보정이
의미를 갖는 건 mid-frequency 발화 룰 (causal_chain_monitor, rate_*) 뿐이다.

Stop-rule (2026-06-08 추가): n ≥ STOP_RULE_MIN_N AND avg_pnl_pct < STOP_RULE_AVG_PNL
→ multiplier 영구 0 (kinetic 이 진입 자체 스킵). 나쁜 룰을 70% 사이즈로 영원히 끌고
가던 문제 해결 (clip(0.7, 1.3) 만으론 은퇴 메커니즘 없었음).

attribution.build_attribution() 의 '룰×전략'(rule|strategy) 실현 성과를 읽어
output/rule_performance.json 에 합성키별 confidence 승수를 기록한다.
kinetic_executor.load_rule_multipliers()/mult_key() 가 이를 읽어 다음 진입 사이징에
반영 → 과거에 돈을 번 (룰,전략)은 크게, 잃은 건 작게 베팅 (실현 성과로 재보정).

같은 룰이라도 스트래들(크기 베팅)과 방향성(방향 베팅)은 *부분 상관* 으로 채점된다:
예) causal_chain_monitor 가 방향은 자주 틀리지만(directional↓) 변동 크기는
맞히면(straddle↑), 두 승수가 다른 방향으로 움직일 수 있다. 다만 **두 readout 은
완전 직교가 아님** — 큰 변동이 음(-) 방향으로 터지면 straddle 익절과 directional
SHORT 도 동시에 맞아 양의 상관을 가짐. "두 배 학습 신호" 는 과장이며 실제는
ρ ∈ (0, 1) 의 부분 상관.

승수 공식
─────────
  표본 부족(< MIN_TRADES)      → 1.0 (조정 안 함)
  factor = min(1, trades / STABLE_N)   (표본 작으면 효과 축소)
  multiplier = clip(1 + avg_pnl_pct × factor, MULT_MIN, MULT_MAX)

  avg_pnl_pct: 프리미엄 대비 평균 실현수익률. 롱 스트래들은 +25% 익절/
  만기 손실 구조라 평균이 룰의 실제 엣지를 잘 대표.

주의: 이 루프는 '사이징(베팅 크기)'만 재보정한다. inference.py의 임계값 T
자체를 바꾸면 어떤 시그널이 발화하는지가 달라지므로(더 위험), 임계값 튜닝은
attribution.json을 참고한 수동 단계로 남겨둔다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUTPUT_DIR
from attribution import build_attribution

RULE_PERF_JSON = OUTPUT_DIR / "rule_performance.json"

# 표본 수 게이트는 전부 *cohort(유효 표본)* 기준 (2026-06-12).
# 같은 (rule|strategy) 로 같은 날 동시 진입한 거래들은 같은 매크로 드라이버로
# 함께 맞고 함께 틀리는 상관 표본 — 거래 수로 세면 금리 한 번 움직임이
# "증거 6개" 로 부풀려져 승수/stop-rule 이 단일 사건에 좌우된다.
MIN_TRADES = 5      # 유효 표본(cohort) 이 미만이면 승수 1.0 고정
STABLE_N   = 15     # 유효 표본 이만큼 쌓이면 승수 효과 100%
MULT_MIN   = 0.7    # clip 범위 좁힘 (이전 0.5 → 0.7)
MULT_MAX   = 1.3    # clip 범위 좁힘 (이전 1.5 → 1.3)

# 나쁜 룰 영구 0 (stop-rule). N건 이상 누적 + 평균 손실이 큰 경우 mult=0 → 진입 스킵.
# kinetic_executor.build_plan 이 multipliers.get(key, 1.0) 로 읽으므로 0이면 scale=0
# → qty 0 처리해야 함. 본 변경에 맞춰 kinetic 도 mult==0 인 경우 진입 자체 스킵.
STOP_RULE_MIN_N      = 10      # 이 이상 누적되어야 stop-rule 평가 시작
STOP_RULE_AVG_PNL    = -0.05   # 평균 P&L% 가 -5% 미만이면 영구 0
KILL_MULT            = 0.0     # 영구 사이즈 0


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def build_feedback(write: bool = True) -> dict:
    report = build_attribution(write=False)
    # 합성키 'rule|strategy' 단위로 승수 생성 — kinetic_executor.mult_key 와 일치.
    # 같은 룰이라도 스트래들(크기 베팅)과 방향성(방향 베팅)은 따로 학습된다.
    by_rule_strategy = report.get("by_rule_strategy", {})

    perf: dict[str, dict] = {}
    for key, s in by_rule_strategy.items():
        n_raw = s["trades"]
        # 유효 표본 = 동시진입 cohort 수. 구버전 attribution.json(cohorts 없음)
        # 폴백은 raw 거래 수 (기존 동작 유지).
        n = s.get("cohorts", n_raw)
        avg = s.get("avg_pnl_pct_eff", s["avg_pnl_pct"])
        killed = False
        if n < MIN_TRADES:
            mult = 1.0
        elif n >= STOP_RULE_MIN_N and avg < STOP_RULE_AVG_PNL:
            # 영구 0 — 충분한 유효 표본에서 평균 손실 명백 → 진입 자체 차단
            mult = KILL_MULT
            killed = True
        else:
            factor = min(1.0, n / STABLE_N)
            mult = _clip(1.0 + avg * factor, MULT_MIN, MULT_MAX)
        perf[key] = {
            "multiplier":      round(mult, 4),
            "killed":          killed,
            "trades":          n_raw,
            "cohorts":         n,
            "win_rate":        s["win_rate"],
            "avg_pnl_pct":     s["avg_pnl_pct"],
            "avg_pnl_pct_eff": avg,
            "total_pnl":       s["total_pnl"],
        }

    if write:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        RULE_PERF_JSON.write_text(
            json.dumps(perf, ensure_ascii=False, indent=2), encoding="utf-8")
    return perf


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    perf = build_feedback()
    print(f"\n{'='*60}")
    print(f"  피드백: 룰별 사이징 승수")
    print(f"{'='*60}")
    if not perf:
        print("\n  완결 거래 없음 — 승수 미생성 (모든 룰 1.0 적용).")
        print(f"\n-> {RULE_PERF_JSON}")
        return

    print(f"  {'rule|strategy':32s} {'mult':>6} {'n':>4} {'n_eff':>5} {'win%':>6} {'effP&L%':>9}")
    print(f"  {'-'*70}")
    for rule, p in sorted(perf.items(), key=lambda kv: -kv[1]["multiplier"]):
        n_eff = p.get("cohorts", p["trades"])
        flag = "" if n_eff >= MIN_TRADES else "  (유효표본부족→1.0)"
        print(f"  {rule:32s} {p['multiplier']:>6.2f} {p['trades']:>4} {n_eff:>5} "
              f"{p['win_rate']*100:>5.0f}% {p['avg_pnl_pct_eff']*100:>8.1f}%{flag}")
    print(f"\n-> {RULE_PERF_JSON}")


if __name__ == "__main__":
    main()

"""
trigger.py - 온톨로지 추론 시그널을 두 가지 액션으로 매핑 (듀얼 전략).

키네틱 레이어의 트리거 소스. vol_monitor z-score 대신 5-레이어 온톨로지
(GARCH→EVT→Copula→Gate→ML)의 추론 룰 출력을 직접 액션으로 연결한다.

두 전략 (같은 추론, 서로 다른 베팅 — 직교 채점용)
─────────────────────────────────────────────
STRADDLE (롱 볼): "큰 변동이 온다"는 크기 베팅. 방향 무관.
  테일/크래시/변동성-리스크 룰만 트리거.
    natural_hedge, crash_vulnerable, co_crash_cluster,
    fat_tail_alert, normal_var_inadequate, causal_chain_monitor
  제외: vol_overpriced(숏볼), rate_*(방향성)

DIRECTIONAL (현물 롱/숏): "이 방향으로 움직인다"는 방향 베팅.
  signal_type이 방향을 가진 룰만 트리거 (MONITOR 제외).
    OVERWEIGHT → +1(롱)   : natural_hedge, rate_beneficiary, vol_overpriced
    UNDERWEIGHT → -1(숏)  : crash_vulnerable, rate_victim
    HEDGE → -1(숏)        : co_crash_cluster, fat_tail_alert(HEDGE)
    MONITOR → 0(방향없음) : normal_var_inadequate, causal_chain_monitor → 제외

두 전략의 룰셋이 다르다는 점이 핵심: rate_*·vol_overpriced 는 스트래들에선
빠지지만 방향성에선 핵심 신호 → 모든 레짐에서 비교 데이터가 쌓인다.

귀인은 (rule × strategy)로 분리되어, 룰마다 "방향을 믿을지 / 크기만 믿을지"를
학습한다.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import OUTPUT_DIR
from ontology.schema import build_empty_graph
from ontology.populate import populate
from ontology.inference import generate_signals


LONG_STRADDLE_RULES: set[str] = {
    # 2026-06-10: 백테스트(n=8, +20.4%) 에서 진짜 알파 확인 + EVT/copula 테제 직결
    "natural_hedge",
    "crash_vulnerable",
    "co_crash_cluster",
    "fat_tail_alert",
    "normal_var_inadequate",
    # causal_chain_monitor / shock_propagator / variance_concentrated 제거 —
    # 정합성 의심 (인과 휴리스틱 또는 PCMCI stale). 라운드 5 비평 선반영.
}

# 롱 스트래들 = 옵션 매수 = IV 지불. vol_overpriced(IV > RV) 인 섹터에서 롱 옵션은
# 구조적으로 마이너스 기대값 → 같은 사이클에 vol_overpriced 가 발화한 섹터의
# STRADDLE 사이즈에 페널티 (숏 볼 트랙은 미구현, 알려진 한계)
SHORT_VOL_PENALTY: float = 0.5

# signal_type → 방향 (+1 롱 / -1 숏 / 0 방향없음)
SIGNAL_DIRECTION: dict[str, int] = {
    "OVERWEIGHT":  +1,
    "UNDERWEIGHT": -1,
    "HEDGE":       -1,   # 테일 증폭/위험 섹터 → 스트레스 시 숏
    "MONITOR":      0,
}

# 방향성 베팅에 부적합한 룰 (논리 mismatch). vol_overpriced 는 "IV 비싸다" 신호인데
# 이를 ETF 현물 롱으로 라우팅하는 건 모순이라 제거 (2026-06-08 비평 반영).
DIRECTIONAL_EXCLUDE_RULES: set[str] = {"vol_overpriced"}

SIGNALS_JSON = OUTPUT_DIR / "ontology_signals.json"
# 사이트 동기화 대상: macro-portal/public/data 가 있으면 함께 발행
PORTAL_DATA_DIR = OUTPUT_DIR.parent.parent / "macro-portal" / "public" / "data"


def _aggregate_straddle(signals: list) -> list[dict]:
    """섹터별 1개로 집계. 같은 섹터 여러 룰 발화 시 최고 confidence 대표.

    vol_overpriced 가 같은 섹터에 발화하면 SHORT_VOL_PENALTY 를 size_penalty 로 부착.
    (롱 옵션이 IV 비싼 섹터에서 구조적으로 마이너스 EV 라는 알려진 한계 보정)
    """
    vol_overpriced_sectors = {
        s.sector for s in signals if s.rule_name == "vol_overpriced"
    }

    by_sector: dict[str, list] = {}
    for s in signals:
        if s.rule_name in LONG_STRADDLE_RULES:
            by_sector.setdefault(s.sector, []).append(s)

    out: list[dict] = []
    for sector, sigs in by_sector.items():
        sigs.sort(key=lambda x: -x.confidence)
        top = sigs[0]
        penalty = SHORT_VOL_PENALTY if sector in vol_overpriced_sectors else 1.0
        out.append({
            "ticker":         sector,
            "confidence":     round(top.confidence, 4),
            "rule":           top.rule_name,
            "signal_type":    top.signal_type.value,
            "regime":         top.regime,
            "all_rules":      [x.rule_name for x in sigs],
            "reasoning":      top.reasoning,
            "size_penalty":   round(penalty, 3),
            "penalty_reason": "vol_overpriced 동시 발화 (IV>RV → 롱 옵션 EV 약화)"
                              if penalty < 1.0 else None,
        })
    out.sort(key=lambda c: -c["confidence"])
    return out


def _aggregate_short_straddle(signals: list) -> list[dict]:
    """숏 스트래들 후보 = vol_overpriced 발화 섹터 ∩ thin_tail_greenlight 발화 섹터.

    *시스템 테제 코히어런스* 조건 — IV 비싸고(vol_overpriced) 동시에 EVT/Copula/CF
    세 안전 측정이 그린라이트(thin_tail_greenlight) 인 섹터만 숏 볼. 라운드 4 비평의
    "꼬리 두꺼운 시스템에서 숏 볼은 자기모순" 직접 반박.

    Returns: [{ticker, confidence, rule, signal_type, regime, reasoning, all_rules}, ...]
    """
    vol_secs = {s.sector for s in signals if s.rule_name == "vol_overpriced"}
    thin_secs = {s.sector for s in signals if s.rule_name == "thin_tail_greenlight"}
    inter = vol_secs & thin_secs
    if not inter:
        return []

    # 섹터별로 두 룰의 SignalNode 를 가져와 reasoning 결합
    by_sector: dict[str, dict] = {}
    for s in signals:
        if s.sector not in inter:
            continue
        if s.rule_name not in ("vol_overpriced", "thin_tail_greenlight"):
            continue
        slot = by_sector.setdefault(s.sector, {"vol": None, "thin": None})
        slot["vol" if s.rule_name == "vol_overpriced" else "thin"] = s

    out: list[dict] = []
    for sector, pair in by_sector.items():
        v, t = pair["vol"], pair["thin"]
        if not (v and t):
            continue
        # 신뢰도: 두 룰의 평균. 둘 다 강하면 강함.
        conf = (float(v.confidence) + float(t.confidence)) / 2.0
        reasoning = list(v.reasoning) + ["─── thin_tail_greenlight ───"] + list(t.reasoning)
        out.append({
            "ticker":      sector,
            "confidence":  round(conf, 4),
            "rule":        "vol_overpriced+thin_tail_greenlight",
            "signal_type": "SHORT_VOL",
            "regime":      v.regime,
            "all_rules":   ["vol_overpriced", "thin_tail_greenlight"],
            "reasoning":   reasoning,
        })
    out.sort(key=lambda c: -c["confidence"])
    return out


def _aggregate_directional(signals: list) -> list[dict]:
    """섹터별 net 방향 집계. 충돌 시 부호있는 confidence 합의 부호로 결정,
    모호하면(net≈0) 제외. 대표 룰=최종 방향과 일치하는 최고 confidence 룰.

    DIRECTIONAL_EXCLUDE_RULES 의 룰은 방향성 라우팅에서 제외 (vol_overpriced 등).
    """
    by_sector: dict[str, list] = {}
    for s in signals:
        if s.rule_name in DIRECTIONAL_EXCLUDE_RULES:
            continue
        if SIGNAL_DIRECTION.get(s.signal_type.value, 0) != 0:
            by_sector.setdefault(s.sector, []).append(s)

    out: list[dict] = []
    for sector, sigs in by_sector.items():
        net = sum(SIGNAL_DIRECTION[s.signal_type.value] * s.confidence for s in sigs)
        if abs(net) < 1e-6:
            continue  # 방향 모호 → 방향성 베팅 스킵
        direction = 1 if net > 0 else -1
        agree = [s for s in sigs if SIGNAL_DIRECTION[s.signal_type.value] == direction]
        agree.sort(key=lambda x: -x.confidence)
        top = agree[0]
        out.append({
            "ticker":      sector,
            "direction":   direction,
            "confidence":  round(top.confidence, 4),
            "rule":        top.rule_name,
            "signal_type": top.signal_type.value,
            "regime":      top.regime,
            "all_rules":   [s.rule_name for s in agree],
            "reasoning":   top.reasoning,
        })
    out.sort(key=lambda c: -c["confidence"])
    return out


def _extract_variance_diagnostics(G, signals: list) -> dict:
    """노드의 Cholesky 분산 분해 + variance_concentrated 시그널 추출 (UI/디버그용)."""
    from ontology.schema import SECTOR_NAMES

    by_sector: list[dict] = []
    for sec in SECTOR_NAMES:
        if sec not in G.nodes:
            continue
        node = G.nodes[sec]
        vd = node.get("var_decomposition", {}) or {}
        if not vd:
            continue
        # 자기 자신 제외 상위 3개 소스
        ext = sorted(
            ((s, w) for s, w in vd.items() if s != sec and w > 0),
            key=lambda kv: -kv[1],
        )[:3]
        ss = node.get("self_share", 0.0)
        ps = node.get("propagation_score", 0.0)
        co = node.get("cholesky_order", -1)
        by_sector.append({
            "ticker":            sec,
            "self_share":        round(float(ss) if ss is not None else 0.0, 4),
            "propagation_score": round(float(ps) if ps is not None else 0.0, 4),
            "cholesky_order":    int(co) if co is not None else -1,
            "top_sources":       [{"src": s, "share": round(w, 4)} for s, w in ext],
        })
    by_sector.sort(key=lambda x: x["cholesky_order"])

    concentrated = []
    for s in signals:
        if s.rule_name != "variance_concentrated":
            continue
        ss = G.nodes[s.sector].get("self_share", 0.0)
        concentrated.append({
            "ticker":     s.sector,
            "self_share": round(float(ss) if ss is not None else 0.0, 4),
            "confidence": round(s.confidence, 4),
            "reasoning":  s.reasoning,
        })
    propagators = []
    for s in signals:
        if s.rule_name != "shock_propagator":
            continue
        ps = G.nodes[s.sector].get("propagation_score", 0.0)
        propagators.append({
            "ticker":            s.sector,
            "propagation_score": round(float(ps) if ps is not None else 0.0, 4),
            "confidence":        round(s.confidence, 4),
            "reasoning":         s.reasoning,
        })

    return {
        "by_sector":             by_sector,
        "variance_concentrated": concentrated,
        "shock_propagators":     propagators,
    }


def get_action_candidates(
    run_all_regimes: bool = False,
    write_json: bool = True,
) -> tuple[list[dict], list[dict], list[dict], str]:
    """온톨로지 파이프라인 1회 실행 →
    (롱 스트래들, 방향성, 숏 스트래들, active_regime).
    """
    G = build_empty_graph()
    active_regime = populate(G)
    signals = generate_signals(G, active_regime, run_all_regimes)

    straddle = _aggregate_straddle(signals)
    directional = _aggregate_directional(signals)
    short_straddle = _aggregate_short_straddle(signals)
    variance_info = _extract_variance_diagnostics(G, signals)

    # NORTA 멀티에셋 시뮬 — 시장 구조만으로 본 21영업일 공동 drawdown 확률
    try:
        from ontology.mvg_mc import run_joint_simulation
        joint_sim = run_joint_simulation(n_sim=5000, horizon=21)
    except Exception as exc:
        print(f"  [WARN] joint simulation 실패: {exc}")
        joint_sim = None

    if write_json:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "active_regime": active_regime,
            "generated":     dt.datetime.now().isoformat(timespec="seconds"),
            "policy": {
                "long_straddle_rules": sorted(LONG_STRADDLE_RULES),
                "signal_direction":    SIGNAL_DIRECTION,
            },
            "straddle_candidates":       straddle,
            "directional_candidates":    directional,
            "short_straddle_candidates": short_straddle,
            "variance_decomposition":    variance_info,
            "joint_simulation":          joint_sim,
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        SIGNALS_JSON.write_text(text, encoding="utf-8")
        # 포털 데이터 폴더가 존재하면 동일 파일을 동기화 (정적 페이지 fetch 대상)
        if PORTAL_DATA_DIR.exists():
            (PORTAL_DATA_DIR / "ontology_signals.json").write_text(text, encoding="utf-8")

    return straddle, directional, short_straddle, active_regime


# 하위호환: 스트래들 후보만 필요할 때
def get_straddle_candidates(
    run_all_regimes: bool = False,
    write_json: bool = True,
) -> tuple[list[dict], str]:
    straddle, _, _, regime = get_action_candidates(run_all_regimes, write_json)
    return straddle, regime


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    straddle, directional, short_straddle, regime = get_action_candidates()
    print(f"\nActive regime: {regime}")

    print(f"\n[LONG STRADDLE] 롱 볼 후보 {len(straddle)}개")
    for c in straddle:
        print(f"  {c['ticker']:5s}  conf={c['confidence']:.2f}  "
              f"[{c['signal_type']}]  rules: {', '.join(c['all_rules'])}")

    print(f"\n[SHORT STRADDLE] 숏 볼 후보 {len(short_straddle)}개 "
          f"(vol_overpriced ∩ thin_tail_greenlight)")
    for c in short_straddle:
        print(f"  {c['ticker']:5s}  conf={c['confidence']:.2f}  "
              f"[{c['signal_type']}]  rules: {', '.join(c['all_rules'])}")

    print(f"\n[DIRECTIONAL] 방향성 후보 {len(directional)}개")
    for c in directional:
        arrow = "LONG " if c["direction"] > 0 else "SHORT"
        print(f"  {c['ticker']:5s}  {arrow}  conf={c['confidence']:.2f}  "
              f"[{c['signal_type']}]  rules: {', '.join(c['all_rules'])}")

    print(f"\n-> {SIGNALS_JSON}")

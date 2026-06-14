"""
inference.py - Rules engine: Object-Link-Action inference with XAI reasoning chains.

Each rule returns a list of SignalNode objects.  All rules carry a 'reasoning' chain
(list of natural-language sentences) that explains exactly why the signal was produced
-- inspired by BondIT's SCORABLE XAI framework.

Rule taxonomy
─────────────
Tail / crash rules
  natural_hedge          : convex to VIX (gamma > 0) → earns when volatility spikes
  crash_vulnerable       : high VIX sensitivity + tail asymmetry → suffers in crashes
  co_crash_cluster       : high lambda_L with many peers → systemic amplifier

Rate rules
  rate_beneficiary       : positive delta to US10Y / US2Y
  rate_victim            : strong negative delta to US10Y

Valuation rules
  vol_overpriced         : VRP > threshold (IV > RV → IV mean-reverts down)
  fat_tail_alert         : GPD xi above threshold → EVT underestimates tail mass
  normal_var_inadequate  : CF/Normal > threshold → normal dist dangerously wrong

Regime-conditional action assembly
  generate_signals       : applies all rules in the current regime and assigns
                           OVERWEIGHT / UNDERWEIGHT / HEDGE / MONITOR labels
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from ontology.schema import (
    EdgeType,
    RegimeType,
    SignalNode,
    SignalType,
    SECTOR_NAMES,
    MACRO_NAMES,
    CAUSAL_MACRO_NAMES,
)


# ---------------------------------------------------------------------------
# Thresholds (all tunable)
# ---------------------------------------------------------------------------

T = {
    # natural_hedge
    "natural_hedge_gamma_min":     0.10,   # gamma(VIX) > 0.10 → meaningful convexity

    # crash_vulnerable
    "crash_delta_vix_max":        -0.008,  # delta(VIX) < -0.008 → sensitive to fear
    "crash_tail_asymm_max":       -0.001,  # tail_asymmetry(VIX) < -0.001 → skewed left

    # rate rules
    "rate_benefit_delta_min":      0.002,  # delta(US10Y or US2Y) > 0.002
    "rate_victim_delta_max":      -0.001,  # delta(US10Y) < -0.001

    # co_crash cluster
    "co_crash_lambda_min":         0.40,   # lambda_L > 0.40 = strong tail co-movement
    "co_crash_peer_count_min":     3,      # at least 3 peers with high lambda_L

    # vol / tail rules
    "vrp_overpriced_min":          0.02,   # VRP > 2pp annualized
    "xi_fat_tail_min":             0.20,   # GPD tail index above 0.20
    "cf_normal_inadequate_min":    2.50,   # CF/Normal ratio above 2.5
    "es99_extreme_min":            0.06,   # ES(99%) > 6% daily loss

    # causal chain propagation
    "chain_delta_min":             0.003,  # |delta| to downstream macro for chain to matter
    "chain_score_min":             0.80,   # min stability score to include in chain
    "chain_max_hops":              4,      # max hops in BFS upstream search

    # Cholesky 분산 분해 룰 (2026-06-10: 룰 제거됐지만 임계값은 보존 — 향후 재도입 가능성)
    "shock_propagator_min":        0.50,
    "variance_concentrated_min":   0.70,

    # thin_tail_greenlight (숏 볼 트랙용, 2026-06-10 신설)
    # 세 안전 측정 동시 통과 → 꼬리 얇음, 숏 볼 안전 가설
    "thin_tail_xi_max":            0.10,   # EVT: ξ < 0.10 = 꼬리 얇음 (정규 비슷)
    "thin_tail_lambda_max":        0.20,   # Copula: max λ_L < 0.20 = 위기 동조 낮음
    "thin_tail_cf_max":            1.50,   # CF/Normal < 1.50 = 정규 VaR 모델 적절
}


# ---------------------------------------------------------------------------
# Helper: pull edge attribute between sector and macro
# ---------------------------------------------------------------------------

def _edge(G: nx.DiGraph, src: str, dst: str, attr: str, default=float("nan")) -> float:
    data = G.get_edge_data(src, dst)
    if data is None:
        return float(default)
    v = data.get(attr, default)
    if v is None:
        return float(default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _node(G: nx.DiGraph, node: str, attr: str, default=float("nan")) -> float:
    v = G.nodes[node].get(attr, default)
    if v is None:
        return float(default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _is_finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Individual rule functions -- each returns list[SignalNode]
# ---------------------------------------------------------------------------

def rule_natural_hedge(G: nx.DiGraph, sector: str) -> list[SignalNode]:
    """Sector is convex to VIX (gamma > threshold) → acts as portfolio hedge."""
    gamma = _edge(G, sector, "VIX", "gamma")
    if not _is_finite(gamma) or gamma <= T["natural_hedge_gamma_min"]:
        return []

    delta   = _edge(G, sector, "VIX", "delta")
    t_gamma = _edge(G, sector, "VIX", "t_gamma")
    speed   = _edge(G, sector, "VIX", "speed")

    reasoning = [
        f"{sector}: gamma(VIX) = {gamma:.4f} > threshold {T['natural_hedge_gamma_min']:.2f} "
        f"(t={t_gamma:.2f}) -- convex payoff to volatility spikes.",
        f"delta(VIX) = {delta:.4f}: "
        + ("linear drag partially offsets convexity."
           if delta < -0.003 else "linear exposure is mild."),
    ]
    if _is_finite(speed) and abs(speed) > 0.01:
        reasoning.append(
            f"3rd-order speed = {speed:.4f} → "
            + ("accelerating downside at large VIX moves." if speed < 0
               else "accelerating upside at large VIX moves.")
        )
    reasoning.append(
        "Rule: natural_hedge. "
        "Sectors with positive gamma to VIX tend to appreciate during fear spikes "
        "because their return function curves upward with volatility -- "
        "a convex hedge in a fear-driven drawdown."
    )

    return [SignalNode(
        signal_id   = str(uuid.uuid4())[:8],
        sector      = sector,
        signal_type = SignalType.OVERWEIGHT,
        regime      = RegimeType.HIGH_VIX.value,
        confidence  = min(0.95, 0.60 + gamma * 0.5),
        rule_name   = "natural_hedge",
        reasoning   = reasoning,
    )]


def rule_crash_vulnerable(G: nx.DiGraph, sector: str) -> list[SignalNode]:
    """Strong negative VIX delta AND left-skewed tail asymmetry."""
    delta_vix = _edge(G, sector, "VIX", "delta")
    tail_asymm = _edge(G, sector, "VIX", "tail_asymmetry")

    if not (_is_finite(delta_vix) and delta_vix < T["crash_delta_vix_max"]):
        return []
    if not (_is_finite(tail_asymm) and tail_asymm < T["crash_tail_asymm_max"]):
        return []

    xi    = _node(G, sector, "xi")
    es99  = _node(G, sector, "es99")

    reasoning = [
        f"{sector}: delta(VIX) = {delta_vix:.4f} < {T['crash_delta_vix_max']:.3f} "
        f"-- strongly negative linear response to fear.",
        f"tail_asymmetry(VIX) = {tail_asymm:.4f} < {T['crash_tail_asymm_max']:.4f} "
        f"-- left tail quantile regression is steeper than right, confirming crash skew.",
    ]
    if _is_finite(xi):
        reasoning.append(f"GPD tail index xi = {xi:.4f} > 0 confirms fat-tailed loss distribution.")
    if _is_finite(es99):
        reasoning.append(f"ES(99%) = {es99:.4f} ({es99*100:.1f}%) -- extreme loss magnitude is significant.")
    reasoning.append(
        "Rule: crash_vulnerable. "
        "Sectors with both high negative linear VIX sensitivity and asymmetric "
        "left-tail quantile response are disproportionately hurt in crash regimes."
    )

    return [SignalNode(
        signal_id   = str(uuid.uuid4())[:8],
        sector      = sector,
        signal_type = SignalType.UNDERWEIGHT,
        regime      = RegimeType.HIGH_VIX.value,
        confidence  = min(0.92, 0.55 + abs(delta_vix) * 30 + abs(tail_asymm) * 200),
        rule_name   = "crash_vulnerable",
        reasoning   = reasoning,
    )]


def rule_co_crash_cluster(G: nx.DiGraph, sector: str) -> list[SignalNode]:
    """High lower tail dependence with many peers → systemic amplifier."""
    high_lambda_peers = [
        nbr for nbr in G.successors(sector)
        if nbr in SECTOR_NAMES
        and G[sector][nbr].get("edge_type") == EdgeType.CO_CRASH_WITH.value
        and _is_finite(G[sector][nbr].get("lambda_lower", float("nan")))
        and G[sector][nbr]["lambda_lower"] >= T["co_crash_lambda_min"]
    ]
    # also check predecessors (undirected pairs stored as directed)
    for nbr in G.predecessors(sector):
        if (nbr in SECTOR_NAMES
                and G[nbr][sector].get("edge_type") == EdgeType.CO_CRASH_WITH.value
                and _is_finite(G[nbr][sector].get("lambda_lower", float("nan")))
                and G[nbr][sector]["lambda_lower"] >= T["co_crash_lambda_min"]
                and nbr not in high_lambda_peers):
            high_lambda_peers.append(nbr)

    if len(high_lambda_peers) < T["co_crash_peer_count_min"]:
        return []

    avg_lam = np.nanmean([
        G[sector][p]["lambda_lower"] if G.has_edge(sector, p)
        else G[p][sector]["lambda_lower"]
        for p in high_lambda_peers
    ])

    reasoning = [
        f"{sector} co-crashes with {len(high_lambda_peers)} peers "
        f"(lambda_L >= {T['co_crash_lambda_min']:.2f}): {', '.join(high_lambda_peers)}.",
        f"Mean lower tail dependence = {avg_lam:.3f} "
        f"-- these sectors jointly decline in extreme drawdowns.",
        "Rule: co_crash_cluster. "
        "High bi-variate tail dependence means diversification fails exactly when "
        "it is needed most.  Sector acts as a systemic amplifier -- adding it increases "
        "portfolio left-tail risk nonlinearly.",
    ]

    return [SignalNode(
        signal_id   = str(uuid.uuid4())[:8],
        sector      = sector,
        signal_type = SignalType.HEDGE,
        regime      = RegimeType.HIGH_VIX.value,
        confidence  = min(0.90, 0.50 + avg_lam * 0.5),
        rule_name   = "co_crash_cluster",
        reasoning   = reasoning,
    )]


def _rate_delta(G: nx.DiGraph, sector: str, macro: str) -> tuple[float, float, float, bool]:
    """(delta, t, gamma, is_ctrl) — partial(통제) 계수 우선, 없으면 이변량 폴백.

    2026-06-12: 이변량 delta(US10Y) 는 유가/시장 교란에 노출 (호르무즈 표본에서
    유가↑→금리↑ 동시 발생 → XLE 가 가짜 '금리 수혜'). partial_results.csv 가
    있으면 OIL·VIX·DXY·SPY 통제 후의 delta_ctrl 로 판정한다.
    """
    d_ctrl = _edge(G, sector, macro, "delta_ctrl")
    if _is_finite(d_ctrl):
        return (d_ctrl,
                _edge(G, sector, macro, "t_delta_ctrl"),
                _edge(G, sector, macro, "gamma_ctrl"),
                True)
    return (_edge(G, sector, macro, "delta"),
            _edge(G, sector, macro, "t_delta"),
            _edge(G, sector, macro, "gamma"),
            False)


def _rate_significant(G: nx.DiGraph, sector: str, macro: str, t: float) -> bool:
    """rate_* 룰의 유의성 게이트 — BH FDR q < 0.10 강제 (감사 P0-1, 옵션1).

    q 부재(직교화 partial 미적용 contrast, 예: US2Y) 시 **발화 금지**. 과거엔
    |t|>1.96 폴백이 있었으나, raw t 단독은 55개 가설 다중비교 미보정이라
    감사(FDR)가 기각한 계수를 US2Y 경유로 뒷문 재발화시켰다 (XLK/XLE 사례).
    q 없는 축은 FDR 통과 불가 = KILLED 로 간주.
    """
    q = _edge(G, sector, macro, "q_delta_ctrl")
    return _is_finite(q) and q < 0.10


def rule_rate_beneficiary(G: nx.DiGraph, sector: str) -> list[SignalNode]:
    """Positive *partial* delta to US10Y or US2Y -- benefits from rising rates.

    2026-06-12 후반: delta 크기 + **t > 1.96 유의성** 동시 요구. 직교화 기저에서
    contrast 복원된 US2Y 는 점추정이 커도 SE 가 부풀어 있어, 크기 게이트만으론
    감사(killed)가 기각한 무의미 계수가 뒷문으로 재발화한다 (XLK/XLE 사례).
    """
    d10, t10, g10, ctrl10 = _rate_delta(G, sector, "US10Y")
    d2,  t2,  g2,  ctrl2  = _rate_delta(G, sector, "US2Y")

    best = None   # (delta, t, gamma, is_ctrl, macro)
    for d, t, g, c, m in [(d10, t10, g10, ctrl10, "US10Y"), (d2, t2, g2, ctrl2, "US2Y")]:
        if (_is_finite(d) and d > T["rate_benefit_delta_min"]
                and _is_finite(t) and t > 0
                and _rate_significant(G, sector, m, t)):
            if best is None or d > best[0]:
                best = (d, t, g, c, m)

    if best is None:
        return []
    best_delta, t_d, gamma, is_ctrl, best_macro = best

    ctrl_note = ("partial -- OIL/VIX/DXY/SPY controlled, rates orthogonalized (lvl+2s10s)"
                 if is_ctrl else "bivariate raw -- partial unavailable, confounding possible")
    gamma_txt = (f"gamma({best_macro}) = {gamma:.4f}: "
                 + ("convex to rate rises, amplifying benefit." if gamma > 0
                    else "concave -- benefit diminishes at larger rate moves.")
                 ) if _is_finite(gamma) else f"gamma({best_macro}) = n/a (contrast-derived coefficient)."
    reasoning = [
        f"{sector}: delta({best_macro}) = {best_delta:.4f} > {T['rate_benefit_delta_min']:.3f} "
        f"(t={t_d:.2f}, {ctrl_note}) -- significantly positive linear rate sensitivity.",
        gamma_txt,
        "Rule: rate_beneficiary. "
        "Positive delta to yield changes implies the sector's returns are positively "
        "correlated with rate increases -- typically financials (NIM expansion) "
        "or sectors with low duration liabilities.",
    ]

    return [SignalNode(
        signal_id   = str(uuid.uuid4())[:8],
        sector      = sector,
        signal_type = SignalType.OVERWEIGHT,
        regime      = RegimeType.MID_VIX.value,
        confidence  = min(0.90, 0.55 + best_delta * 60),
        rule_name   = "rate_beneficiary",
        reasoning   = reasoning,
    )]


def rule_rate_victim(G: nx.DiGraph, sector: str) -> list[SignalNode]:
    """Strongly negative *partial* delta to US10Y -- suffers from rate rises.

    delta 크기 + t < -1.96 유의성 동시 요구 (rate_beneficiary 와 동일 근거).
    """
    d10, t_d, gamma, is_ctrl = _rate_delta(G, sector, "US10Y")
    if not (_is_finite(d10) and d10 < T["rate_victim_delta_max"]
            and _is_finite(t_d) and t_d < 0
            and _rate_significant(G, sector, "US10Y", t_d)):
        return []

    speed = _edge(G, sector, "US10Y", "speed")

    ctrl_note = ("partial -- OIL/VIX/DXY/SPY controlled"
                 if is_ctrl else "bivariate raw -- partial unavailable, confounding possible")
    reasoning = [
        f"{sector}: delta(US10Y) = {d10:.4f} < {T['rate_victim_delta_max']:.3f} "
        f"(t={t_d:.2f}, {ctrl_note}) -- significant negative linear rate sensitivity.",
        f"gamma(US10Y) = {gamma:.4f}: "
        + ("concave to rate rises -- losses accelerate as rates increase further." if gamma < 0
           else "convex -- some protection at large moves."),
    ]
    if _is_finite(speed) and speed < -0.01:
        reasoning.append(
            f"3rd-order speed = {speed:.4f} → "
            "accelerating pain at extreme rate rise scenarios (speed < 0)."
        )
    reasoning.append(
        "Rule: rate_victim. "
        "Long-duration sectors (utilities, real estate) have equity duration "
        "analogous to bonds -- rising rates discount future cash flows at a higher rate, "
        "compressing P/E multiples and increasing financing costs."
    )

    return [SignalNode(
        signal_id   = str(uuid.uuid4())[:8],
        sector      = sector,
        signal_type = SignalType.UNDERWEIGHT,
        regime      = RegimeType.MID_VIX.value,
        confidence  = min(0.90, 0.55 + abs(d10) * 60),
        rule_name   = "rate_victim",
        reasoning   = reasoning,
    )]


def rule_vol_overpriced(G: nx.DiGraph, sector: str) -> list[SignalNode]:
    """진짜 VRP > threshold → 옵션 IV(Q) 가 실현 변동성(P) 보다 비쌈 → 숏 볼 후보.

    2026-06-08 수정: 기존 vrp_latest(=σ_GARCH−RV) 는 *물리측도* 모델 예측의 잔차일 뿐
    옵션 가격 정보가 없음 — VRP 정의에 반함. 본 룰은 *vrp_iv* (= ATM_IV − RV_20d) 만 본다.
    vrp_iv 가 NaN(휴장/API 실패) 이면 발화 안 함 (안전한 silence).
    """
    vrp = _node(G, sector, "vrp_iv")
    if not (_is_finite(vrp) and vrp > T["vrp_overpriced_min"]):
        return []

    iv_atm = _node(G, sector, "iv_atm")

    reasoning = [
        f"{sector}: VRP_true = ATM_IV - RV_20d = {vrp*100:.2f}pp "
        f"> threshold {T['vrp_overpriced_min']*100:.0f}pp.",
        f"ATM_IV = {iv_atm*100:.1f}% 연환산 (콜+풋 평균, Alpaca options snapshot).",
        "옵션 시장이 실현 변동성보다 높은 변동성에 값을 매기고 있음 (Q > P).",
        "Rule: vol_overpriced. "
        "지속적 양의 VRP 는 *꼬리 리스크에 대한 보상* — 공짜 알파 아님. "
        "숏 볼 트랙 미구현 상태에서 본 신호는 (a) STRADDLE 사이즈 ×0.5 페널티, "
        "(b) DIRECTIONAL 라우팅 제외 로 사용 (trigger.py).",
    ]

    return [SignalNode(
        signal_id   = str(uuid.uuid4())[:8],
        sector      = sector,
        signal_type = SignalType.OVERWEIGHT,
        regime      = RegimeType.LOW_VIX.value,
        confidence  = min(0.88, 0.55 + vrp * 6),
        rule_name   = "vol_overpriced",
        reasoning   = reasoning,
    )]


def rule_fat_tail_alert(G: nx.DiGraph, sector: str) -> list[SignalNode]:
    """GPD xi above threshold -- EVT analysis flags fat-tailed distribution."""
    xi   = _node(G, sector, "xi")
    es99 = _node(G, sector, "es99")
    cf_n = _node(G, sector, "cf_over_normal")

    if not (_is_finite(xi) and xi > T["xi_fat_tail_min"]):
        return []

    reasoning = [
        f"{sector}: GPD tail index xi = {xi:.4f} > {T['xi_fat_tail_min']:.2f} "
        f"-- Generalized Pareto Distribution fit confirms heavy-tailed loss process.",
    ]
    if _is_finite(es99):
        reasoning.append(f"EVT Expected Shortfall (99%) = {es99*100:.2f}% per day "
                         "-- tail losses are severe when triggered.")
    if _is_finite(cf_n) and cf_n > T["cf_normal_inadequate_min"]:
        reasoning.append(
            f"Cornish-Fisher VaR is {cf_n:.2f}x normal VaR -- "
            "standard normal-assumption risk models dangerously underestimate tail exposure."
        )
    reasoning.append(
        "Rule: fat_tail_alert. "
        "Fat-tailed distributions (xi > 0) imply tail losses are power-law rather than "
        "exponential -- no finite variance if xi >= 0.5, no finite mean if xi >= 1.0. "
        "Standard VaR models based on normality are structurally inadequate for this sector."
    )

    signal_type = (SignalType.HEDGE if xi > 0.35 else SignalType.MONITOR)
    return [SignalNode(
        signal_id   = str(uuid.uuid4())[:8],
        sector      = sector,
        signal_type = signal_type,
        regime      = RegimeType.HIGH_VIX.value,
        confidence  = min(0.92, 0.55 + xi * 0.8),
        rule_name   = "fat_tail_alert",
        reasoning   = reasoning,
    )]


def rule_normal_var_inadequate(G: nx.DiGraph, sector: str) -> list[SignalNode]:
    """CF/Normal VaR ratio above threshold -- normal dist dangerously wrong."""
    cf_n = _node(G, sector, "cf_over_normal")
    skew = _node(G, sector, "skewness")
    kurt = _node(G, sector, "kurtosis")

    if not (_is_finite(cf_n) and cf_n > T["cf_normal_inadequate_min"]):
        return []

    reasoning = [
        f"{sector}: CF VaR(99%) / Normal VaR(99%) = {cf_n:.2f}x "
        f"> threshold {T['cf_normal_inadequate_min']:.1f}x.",
    ]
    if _is_finite(skew):
        reasoning.append(f"Return skewness = {skew:.3f} "
                         + ("(left-skewed -- negative surprises dominate)." if skew < 0
                            else "(right-skewed)."))
    if _is_finite(kurt):
        reasoning.append(f"Excess kurtosis = {kurt:.3f} "
                         + ("-- significant leptokurtosis (fat tails)." if kurt > 1 else "."))
    reasoning.append(
        "Rule: normal_var_inadequate. "
        "Cornish-Fisher expansion adjusts normal quantiles for observed skewness "
        "and kurtosis.  A ratio > 2.5 means institutions using normal-assumption "
        "VaR models are holding less than half the capital actually needed to cover "
        "99% tail losses in this sector."
    )

    return [SignalNode(
        signal_id   = str(uuid.uuid4())[:8],
        sector      = sector,
        signal_type = SignalType.MONITOR,
        regime      = RegimeType.MID_VIX.value,
        confidence  = min(0.90, 0.50 + (cf_n - 2.5) * 0.15),
        rule_name   = "normal_var_inadequate",
        reasoning   = reasoning,
    )]


# ---------------------------------------------------------------------------
# Cholesky variance decomposition rules (mvg_mc)
# ---------------------------------------------------------------------------

def rule_shock_propagator(G: nx.DiGraph, sector: str) -> list[SignalNode]:
    """Shapley-Owen 평균 분산 분해에서 본 섹터의 잔차가 다른 섹터들 분산 합에 가장
    크게 등장하는 케이스 (M=200 무작위 ordering 평균).

    *주의 — 인과 아님*: Shapley 는 *공정 배분* 의 비인과 공식. ordering 의존성(=Cholesky
    의 임의 휴리스틱)을 알고리즘 레벨에서 제거했지만, "한계 기여 평균" 은 여전히 상관
    구조 통계일 뿐 인과가 아니다. PCMCI 인과 체인은 매크로 변수에만 적용 가능 —
    섹터 cross-section 에는 미적용.
    """
    prop = _node(G, sector, "propagation_score")
    if not (_is_finite(prop) and prop >= T["shock_propagator_min"]):
        return []

    self_share = _node(G, sector, "self_share")
    chol_order = _node(G, sector, "cholesky_order")

    # 가장 크게 영향을 받는 상위 섹터 3개 (콜레스키 순서 조건부)
    downstream: list[tuple[str, float]] = []
    for nbr in G.successors(sector):
        if nbr in SECTOR_NAMES and nbr != sector:
            data = G[sector][nbr]
            if data.get("edge_type") == EdgeType.CO_MOVES_WITH.value:
                downstream.append((nbr, float(data.get("attribution", 0) or 0)))
    downstream.sort(key=lambda kv: -kv[1])
    top3 = downstream[:3]

    reasoning = [
        f"{sector}: propagation_score(Shapley) = {prop:.3f} "
        f"≥ {T['shock_propagator_min']:.2f}.",
        f"self_share = {self_share:.3f} (Shapley 평균, 200 ordering).",
    ]
    if top3:
        ds_str = ", ".join(f"{t}({a*100:.1f}%)" for t, a in top3)
        reasoning.append(f"평균 분해의 상위 하류: {ds_str}.")
    reasoning.append(
        "Rule: shock_propagator. "
        "*인과 아님*: Shapley-Owen 공정 배분의 한계 기여 평균 (200 무작위 ordering). "
        "ordering 휴리스틱은 제거됐지만 여전히 상관 구조 통계 — '스트레스 사건의 진앙' "
        "은 *prior* 로만 사용. 변동 크기(스트래들)와 약세 방향(현물 숏) 보조 입력."
    )

    return [SignalNode(
        signal_id   = str(uuid.uuid4())[:8],
        sector      = sector,
        signal_type = SignalType.HEDGE,
        regime      = RegimeType.HIGH_VIX.value,
        confidence  = min(0.90, 0.55 + min(prop, 2.0) * 0.15),
        rule_name   = "shock_propagator",
        reasoning   = reasoning,
    )]


def rule_variance_concentrated(G: nx.DiGraph, sector: str) -> list[SignalNode]:
    """Shapley-Owen self_share 가 높음 → 다른 섹터 충격에 둔감 (해석).

    *주의 — 인과 아님*: Shapley 는 ordering 의존성을 제거하지만, "self_share 가
    높다 → 다른 섹터 충격에 둔감" 은 여전히 *상관 휴리스틱*. 인과 디커플링을
    보장하지 않으며, 산업-특화 카탈리스트 노출 가능성의 *prior* 로만 사용.
    """
    self_share = _node(G, sector, "self_share")
    if not (_is_finite(self_share) and self_share >= T["variance_concentrated_min"]):
        return []

    prop = _node(G, sector, "propagation_score")

    reasoning = [
        f"{sector}: self_share(Shapley) = {self_share:.3f} "
        f"≥ {T['variance_concentrated_min']:.2f}.",
        f"propagation_score = {prop:.3f} (낮을수록 다른 섹터 분산에 적게 등장).",
        "Rule: variance_concentrated. "
        "*인과 아님*: Shapley 평균 분해의 자기 기여 비율. 'self_share 가 높다 → "
        "디커플링' 은 가우시안 가정 + 상관 해석. 산업-특화 카탈리스트 노출 가능성의 "
        "prior 로만 사용.",
    ]

    return [SignalNode(
        signal_id   = str(uuid.uuid4())[:8],
        sector      = sector,
        signal_type = SignalType.MONITOR,
        regime      = RegimeType.MID_VIX.value,
        confidence  = min(0.85, 0.50 + (self_share - T["variance_concentrated_min"]) * 1.0),
        rule_name   = "variance_concentrated",
        reasoning   = reasoning,
    )]


# ---------------------------------------------------------------------------
# Causal chain propagation helpers + rule
# ---------------------------------------------------------------------------

def _upstream_chains(
    G: nx.DiGraph,
    target: str,
    min_score: float,
    max_hops: int,
) -> list[tuple[list[str], int, float]]:
    """
    BFS upstream via TRANSMITS_TO edges from `target`.

    반환: [(path, total_lag_months, min_score_along_path), ...]
      path = [upstream_root, ..., intermediate, target]
      total_lag = 체인의 총 지연 개월 합산
      min_score = 체인 내 가장 낮은 stability score
    """
    from collections import deque
    results: list[tuple[list[str], int, float]] = []
    # queue: (current_node, path_so_far, accumulated_lag, min_sc)
    queue: deque = deque([(target, [target], 0, 1.0)])
    visited: set[str] = {target}

    while queue:
        node, path, lag, min_sc = queue.popleft()
        if len(path) > max_hops + 1:
            continue

        for pred in G.predecessors(node):
            data = G[pred][node]
            if data.get("edge_type") != EdgeType.TRANSMITS_TO.value:
                continue
            if data.get("stability", "historical") == "historical":
                continue
            score = float(data.get("score", 0) or 0)
            if score < min_score:
                continue

            new_path = [pred] + path
            new_lag  = lag + int(data.get("lag_months", 0) or 0)
            new_min  = min(min_sc, score)

            # 최소 1hop 이상이어야 결과에 포함
            if len(new_path) >= 2:
                results.append((new_path, new_lag, new_min))

            if pred not in visited:
                visited.add(pred)
                queue.append((pred, new_path, new_lag, new_min))

    # 가장 긴 체인(hop 수 많은) 순으로 정렬, 동일 길이면 score 높은 것 우선
    results.sort(key=lambda x: (-len(x[0]), -x[2]))
    return results


def rule_causal_chain_monitor(G: nx.DiGraph, sector: str) -> list[SignalNode]:
    """
    PCMCI 인과 체인에서 이 섹터에 도달하는 업스트림 충격 경로를 추적한다.

    섹터가 SENSITIVE_TO 관계를 맺고 있는 매크로 변수(VIX, US10Y, US2Y, DXY)가
    TRANSMITS_TO 체인의 종점에 있을 때, 그 업스트림 전파 경로를 reasoning에 포함한
    MONITOR 시그널을 발생시킨다.

    예시: XLK → SENSITIVE_TO → VIX ← FED_FUNDS(3m) ← US2Y(1m) ← CPI(3m)
    해석: "CPI 충격 → 7개월 후 VIX 영향 → XLK 리스크 노출"
    """
    _ALL_MACRO = {**MACRO_NAMES, **CAUSAL_MACRO_NAMES}
    signals: list[SignalNode] = []
    best_chains: list[tuple[str, list[str], int, float, float]] = []
    # (macro, path, total_lag, min_score, |delta|)

    for macro in MACRO_NAMES:
        delta = _edge(G, sector, macro, "delta")
        if not _is_finite(delta) or abs(delta) < T["chain_delta_min"]:
            continue

        chains = _upstream_chains(
            G, macro,
            min_score=T["chain_score_min"],
            max_hops=T["chain_max_hops"],
        )
        if not chains:
            continue

        # 해당 macro에서 가장 의미있는 체인 1개만 사용 (가장 길고 score 높은 것)
        best_path, total_lag, min_sc = chains[0]
        best_chains.append((macro, best_path, total_lag, min_sc, abs(delta)))

    if not best_chains:
        return []

    # 신뢰도: (min_score × |delta|)가 가장 높은 체인 기준
    best_chains.sort(key=lambda x: -(x[3] * x[4]))
    top_macro, top_path, top_lag, top_min_sc, top_delta = best_chains[0]

    # 체인 문자열 구성
    def _chain_str(path: list[str], end_macro: str) -> str:
        labels = []
        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            data = G.get_edge_data(src, dst) or {}
            lag  = int(data.get("lag_months", 0) or 0)
            coef = float(data.get("coef", 0) or 0)
            stab = data.get("stability", "?")
            sign = "+" if coef >= 0 else "-"
            labels.append(f"{src} --({sign}{lag}m,{stab[:4]})--> {dst}")
        return "  =>  ".join(labels)

    reasoning: list[str] = [
        f"{sector}: {top_macro}에 delta={top_delta:.4f}로 노출 -- 업스트림 인과 체인 감지.",
        f"전파 경로: {_chain_str(top_path, top_macro)}",
        f"총 지연: ~{top_lag}개월 (체인 내 최소 stability_score={top_min_sc:.3f})",
    ]

    # 추가 체인이 있으면 간략히 열거
    if len(best_chains) > 1:
        for mac, path, lag, sc, d in best_chains[1:3]:
            reasoning.append(
                f"보조 체인: {path[0]} --> ... --> {mac} "
                f"(총 {lag}m, score={sc:.3f}, delta={d:.4f})"
            )

    reasoning.append(
        "Rule: causal_chain_monitor. "
        "PCMCI 인과 발견(2022 레짐 기준)에서 검증된 전파 경로. "
        "업스트림 변수(OIL/CPI/FED_FUNDS 등) 이상 시 해당 섹터까지 "
        f"~{top_lag}개월 내 영향 전달 가능."
    )

    confidence = min(0.88, 0.50 + top_min_sc * top_delta * 15)

    return [SignalNode(
        signal_id   = str(uuid.uuid4())[:8],
        sector      = sector,
        signal_type = SignalType.MONITOR,
        regime      = RegimeType.MID_VIX.value,
        confidence  = confidence,
        rule_name   = "causal_chain_monitor",
        reasoning   = reasoning,
    )]


# ---------------------------------------------------------------------------
# All rules registry
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Thin-tail greenlight (숏 볼 트랙 전용, 2026-06-10 신설)
# ---------------------------------------------------------------------------

def rule_thin_tail_greenlight(G: nx.DiGraph, sector: str) -> list[SignalNode]:
    """EVT + Copula + Cornish-Fisher 세 안전 측정 동시 통과 → 숏 볼 안전 가설.

    2026-06-12 (라운드 6 비평 4번): 판정을 점추정이 아니라 **부트스트랩 CI 상한**
    으로 한다. ~370일 표본에서 ξ(SE≈0.2)·λ_L(SE>0.1) 점추정은 노이즈와 구분 불가
    — 이 룰은 *무한손실 숏 스트래들* 의 게이트라 "노이즈가 안전을 선언" 하는
    비대칭 위험이 가장 크다. CI 상한이 없으면(구버전 캐시) 발화하지 않는다 (안전 폐쇄).

    조건 (전부 95% 상한 기준):
      xi_hi < 0.10              (EVT: 최악으로 봐도 꼬리 얇음)
      max λ_L_hi 인접 < 0.20    (Copula: 최악으로 봐도 위기 동조 낮음)
      CF/Normal_hi < 1.50       (CF: 최악으로 봐도 정규 VaR 적절)

    단독으론 *정보 신호* (SignalType.MONITOR). trigger 단에서 vol_overpriced 와
    AND 조건으로 만나면 short_straddle 후보가 됨 — 시스템 테제 코히어런스.
    """
    xi_pt = _node(G, sector, "xi")
    xi    = _node(G, sector, "xi_hi")
    cf_pt = _node(G, sector, "cf_over_normal")
    cf_n  = _node(G, sector, "cf_over_normal_hi")

    # CI 상한 부재 (구버전 tail_gpd.csv 등) → 안전 폐쇄: 발화 안 함
    if not _is_finite(xi) or not _is_finite(cf_n):
        return []

    # 가장 높은 λ_L 인접 — 상한 기준 (상한 부재 페어는 점추정의 1.5배로 보수 대체)
    lambda_l_max = 0.0
    for nbr in list(G.successors(sector)) + list(G.predecessors(sector)):
        if nbr not in SECTOR_NAMES or nbr == sector:
            continue
        data = G.get_edge_data(sector, nbr) or G.get_edge_data(nbr, sector) or {}
        if data.get("edge_type") != EdgeType.CO_CRASH_WITH.value:
            continue
        lam_hi = data.get("lambda_lower_hi")
        if not _is_finite(lam_hi):
            lam_pt = data.get("lambda_lower", 0)
            lam_hi = float(lam_pt) * 1.5 if _is_finite(lam_pt) else 0.0
        lambda_l_max = max(lambda_l_max, float(lam_hi))

    if xi >= T["thin_tail_xi_max"]:
        return []
    if lambda_l_max > T["thin_tail_lambda_max"]:
        return []
    if cf_n > T["thin_tail_cf_max"]:
        return []

    reasoning = [
        f"{sector}: ξ 95%상한 = {xi:.3f} < {T['thin_tail_xi_max']:.2f} "
        f"(점추정 {xi_pt:.3f}) → 최악으로 봐도 꼬리 얇음 (EVT, 부트스트랩).",
        f"max λ_L 인접 95%상한 = {lambda_l_max:.3f} < {T['thin_tail_lambda_max']:.2f} → "
        f"최악으로 봐도 위기 동조 낮음 (Copula).",
        f"CF/Normal 95%상한 = {cf_n:.2f} < {T['thin_tail_cf_max']:.2f} "
        f"(점추정 {cf_pt:.2f}) → 정규 VaR 적절.",
        "Rule: thin_tail_greenlight. "
        "세 안전 측정의 *CI 상한* 동시 통과 → '꼬리가 얇다' 가설. 점추정 게이트는 "
        "표본 노이즈가 무한손실 방향의 안전을 선언할 수 있어 상한으로 보수화 (06-12). "
        "vol_overpriced 와 AND 발화 시 숏 스트래들 후보.",
    ]

    # 신뢰도: 세 안전 측정(상한)이 임계의 안쪽에 있을수록 높음
    safety = (1 - xi / T["thin_tail_xi_max"]) * 0.4 \
           + (1 - lambda_l_max / T["thin_tail_lambda_max"]) * 0.3 \
           + (1 - cf_n / T["thin_tail_cf_max"]) * 0.3
    confidence = min(0.85, 0.55 + max(0.0, safety) * 0.30)

    return [SignalNode(
        signal_id   = str(uuid.uuid4())[:8],
        sector      = sector,
        signal_type = SignalType.MONITOR,
        regime      = RegimeType.LOW_VIX.value,
        confidence  = confidence,
        rule_name   = "thin_tail_greenlight",
        reasoning   = reasoning,
    )]


# 2026-06-10: 정합성 의심 룰 제거 (라운드 5 비평 선반영).
# - shock_propagator / variance_concentrated: Shapley 평균이라도 비인과 휴리스틱.
#   백테스트(123일, n=50) 결과 -3.9% / -4.3% 누적 손실. 시스템 테제와 직결 X.
# - causal_chain_monitor: PCMCI 4개월 stale + mid_vix 거의 모든 섹터 발화 = 변별력
#   낮음. PCMCI 재실행 후 재평가까지 비활성.
# Cholesky/Shapley 분산 분해는 *분석 도구*(UI 표시)로만 유지 — inference 룰 X.

ALL_RULES = [
    rule_natural_hedge,
    rule_crash_vulnerable,
    rule_co_crash_cluster,
    rule_rate_beneficiary,
    rule_rate_victim,
    rule_vol_overpriced,
    rule_fat_tail_alert,
    rule_normal_var_inadequate,
    rule_thin_tail_greenlight,   # 숏 볼 트랙용 (2026-06-10)
]

# Regime-rule filter: only emit signals applicable to the active regime
# (rules can fire in multiple regimes)
REGIME_RULES: dict[str, list] = {
    RegimeType.LOW_VIX.value: [
        rule_vol_overpriced,
        rule_rate_beneficiary,
        rule_thin_tail_greenlight,    # 평상시에만 — high_vix 에선 자동 비활성
    ],
    RegimeType.MID_VIX.value: [
        rule_rate_beneficiary,
        rule_rate_victim,
        rule_normal_var_inadequate,
        rule_vol_overpriced,
        rule_thin_tail_greenlight,
    ],
    RegimeType.HIGH_VIX.value: [
        rule_natural_hedge,
        rule_crash_vulnerable,
        rule_co_crash_cluster,
        rule_fat_tail_alert,
        rule_rate_victim,
        # thin_tail_greenlight 제외 — high_vix 에선 숏 볼 위험 (시스템 가설)
    ],
}


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def generate_signals(
    G: nx.DiGraph,
    active_regime: str,
    run_all_regimes: bool = False,
) -> list[SignalNode]:
    """
    Apply regime-conditional rules to all sectors and return Signal nodes.

    Parameters
    ----------
    G              : populated ontology graph
    active_regime  : current regime string (from populate.py)
    run_all_regimes: if True, fire ALL rules regardless of regime (research mode)
    """
    rules = ALL_RULES if run_all_regimes else REGIME_RULES.get(active_regime, ALL_RULES)
    signals: list[SignalNode] = []

    for sector in SECTOR_NAMES:
        for rule_fn in rules:
            try:
                new_sigs = rule_fn(G, sector)
                signals.extend(new_sigs)
            except Exception as exc:
                print(f"  [WARN] rule {rule_fn.__name__} failed on {sector}: {exc}")

    # Attach signal nodes to graph
    for sig in signals:
        G.add_node(sig.signal_id, **sig.to_dict())
        G.add_edge(sig.sector, sig.signal_id, edge_type="GENERATES")

    return signals


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def signals_to_dataframe(signals: list[SignalNode]) -> "pd.DataFrame":
    import pandas as pd
    rows = []
    for s in signals:
        rows.append({
            "sector":      s.sector,
            "signal":      s.signal_type.value,
            "rule":        s.rule_name,
            "regime":      s.regime,
            "confidence":  round(s.confidence, 3),
            "reasoning":   " | ".join(s.reasoning),
        })
    return pd.DataFrame(rows).sort_values(["sector", "confidence"], ascending=[True, False])


def print_signals(signals: list[SignalNode]) -> None:
    from ontology.schema import SECTOR_NAMES
    grouped: dict[str, list] = {}
    for s in signals:
        grouped.setdefault(s.signal_type.value, []).append(s)

    order = [SignalType.OVERWEIGHT.value, SignalType.UNDERWEIGHT.value,
             SignalType.HEDGE.value, SignalType.MONITOR.value]

    for sig_type in order:
        sigs = grouped.get(sig_type, [])
        if not sigs:
            continue
        print(f"\n{'='*64}")
        print(f"  {sig_type}  ({len(sigs)} signals)")
        print(f"{'='*64}")
        for s in sorted(sigs, key=lambda x: -x.confidence):
            name = SECTOR_NAMES.get(s.sector, s.sector)
            print(f"\n  {s.sector} ({name})  conf={s.confidence:.2f}  rule={s.rule_name}")
            for i, line in enumerate(s.reasoning, 1):
                print(f"    [{i}] {line}")

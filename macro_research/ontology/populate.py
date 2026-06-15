"""
populate.py - Load all computed research results and populate the ontology graph.

Data sources:
  output/static_results.csv        -> delta, gamma, t-stats, R² per (sector, macro)
  output/poly4_static.csv          -> speed, color per (sector, macro)
  output/tail_gpd.csv              -> xi, var99, es99, u per sector
  output/tail_dep_sector_lower.csv -> lambda_L between sectors
  output/tail_dep_macro_lower.csv  -> lambda_L sector vs macro
  output/tail_asymmetry.csv        -> tail_asymmetry per (sector, macro)
  output/vrp_capm_series.csv       -> latest VRP per sector
  output/tail_cf_var.csv           -> cf_var99, cf_over_normal, skewness, kurtosis
  output/macro_levels.parquet      -> current VIX level -> active regime
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

# Allow imports from parent directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import OUTPUT_DIR
from ontology.schema import (
    build_empty_graph,
    EdgeType,
    RegimeType,
    SensitiveToEdge,
    CoCrashEdge,
    CoMovesEdge,
    TransmitsToEdge,
    SECTOR_NAMES,
    MACRO_NAMES,
    CAUSAL_MACRO_NAMES,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _safe(val) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _load_csv_safe(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  [WARN] not found: {path.name}")
        return None
    return pd.read_csv(path, index_col=0)


# ---------------------------------------------------------------------------
# Step 1: sector-level tail risk (GPD + CF VaR)
# ---------------------------------------------------------------------------

def _populate_tail_risk(G: nx.DiGraph) -> None:
    gpd = _load_csv_safe(OUTPUT_DIR / "tail_gpd.csv")
    cf  = _load_csv_safe(OUTPUT_DIR / "tail_cf_var.csv")

    for sec in SECTOR_NAMES:
        attrs: dict = {}
        if gpd is not None and sec in gpd.index:
            row = gpd.loc[sec]
            attrs.update({
                "xi":    _safe(row.get("xi")),
                "xi_lo": _safe(row.get("xi_lo")),   # 부트스트랩 90% CI (2026-06-12)
                "xi_hi": _safe(row.get("xi_hi")),
                "var99": _safe(row.get("var99")),
                "es99":  _safe(row.get("es99")),
            })
        if cf is not None and sec in cf.index:
            row = cf.loc[sec]
            attrs.update({
                "cf_var99":          _safe(row.get("cf_var99")),
                "cf_over_normal":    _safe(row.get("cf_over_normal")),
                "cf_over_normal_hi": _safe(row.get("cf_over_normal_hi")),
                "skewness":          _safe(row.get("skewness")),
                "kurtosis":          _safe(row.get("kurtosis")),
            })
        if attrs:
            nx.set_node_attributes(G, {sec: attrs})

    print("  [OK] tail risk attributes populated")


# ---------------------------------------------------------------------------
# Step 2: VRP latest snapshot
# ---------------------------------------------------------------------------

def _populate_vrp(G: nx.DiGraph) -> None:
    vrp_path = OUTPUT_DIR / "vrp_capm_series.csv"
    if not vrp_path.exists():
        print("  [WARN] vrp_capm_series.csv not found, skipping VRP")
        return

    vrp_df = pd.read_csv(vrp_path, index_col=0, parse_dates=True)
    latest = vrp_df.dropna(how="all").tail(1)
    for sec in SECTOR_NAMES:
        if sec in latest.columns:
            val = _safe(latest[sec].iloc[0])
            nx.set_node_attributes(G, {sec: {"vrp_latest": val}})
    print("  [OK] (legacy) σ_GARCH 기반 VRP snapshot populated")


def _populate_vrp_iv(G: nx.DiGraph) -> None:
    """ATM 콜+풋 옵션에서 추출한 *진짜* IV 로 VRP 재계산.

    VRP_true = ATM_IV - RV_20d (Q vs P 갭). 휴장/API 실패 시 NaN — rule_vol_overpriced
    가 NaN 인 경우 발화 안 함 (즉 휴장 중엔 vol_overpriced 신호 자동 비활성).
    """
    # RV_20d 계산 (sector_returns.parquet)
    ret_path = OUTPUT_DIR / "sector_returns.parquet"
    if not ret_path.exists():
        print("  [WARN] sector_returns.parquet 없음, VRP_IV 스킵")
        return
    rets = pd.read_parquet(ret_path)
    rv_annual = rets.tail(20).std() * np.sqrt(252.0)

    try:
        from iv_extractor import get_atm_iv, append_history
    except Exception as exc:
        print(f"  [WARN] iv_extractor import 실패: {exc}")
        return

    count = 0
    for sec in SECTOR_NAMES:
        if sec not in rv_annual.index:
            continue
        try:
            info = get_atm_iv(sec)
        except Exception as exc:
            print(f"  [WARN] {sec} ATM IV 조회 실패: {exc}")
            info = None
        if not info:
            continue
        rv = float(rv_annual[sec])
        iv = float(info["iv"])
        vrp_iv = iv - rv
        nx.set_node_attributes(G, {sec: {"iv_atm": round(iv, 4),
                                          "vrp_iv": round(vrp_iv, 4)}})
        # 시계열 누적 — 매시간 cron 이 한 행 append
        try:
            append_history(sec, info, rv, vrp_iv)
        except Exception as exc:
            print(f"  [WARN] {sec} IV history append 실패: {exc}")
        count += 1
    print(f"  [OK] VRP_IV 부착: {count}/{len(SECTOR_NAMES)} 섹터 + "
          f"iv_history.jsonl append")


# realized-vol z-score (event_vol 룰 입력, 라운드9) — vol_monitor_pipeline.py 와 동일 공식
RV_Z_LOOKBACK = 252   # 기준선 창 (1년)
RV_Z_RECENT   = 10    # 최근 RV 창 (10일)


def compute_rv_zscore(rets: "pd.Series") -> float:
    """단일 섹터 실현변동성 z-score. vol_monitor_pipeline 공식과 동일:
    RV = rolling(10).std()*√252; z = (current_RV − baseline.mean) / baseline.std,
    baseline = RV_series[-252:-10]. 데이터 부족/비유한 시 NaN.
    *VIX 와 독립적인 실현 변동성 급등* 측정 (백테스트도 동일 함수로 재현 가능)."""
    r = rets.dropna()
    if len(r) < RV_Z_LOOKBACK // 2:
        return float("nan")
    rv_series  = r.rolling(RV_Z_RECENT).std() * np.sqrt(252.0)
    current_rv = float(rv_series.iloc[-1])
    hist_rv    = rv_series.iloc[-RV_Z_LOOKBACK:-RV_Z_RECENT]
    baseline   = float(hist_rv.mean())
    std_rv     = float(hist_rv.std()) if hist_rv.std() > 0 else 1e-6
    if not (np.isfinite(current_rv) and np.isfinite(baseline)):
        return float("nan")
    return (current_rv - baseline) / std_rv


def _populate_rv_zscore(G: nx.DiGraph) -> None:
    """섹터별 실현변동성 z-score 를 rv_zscore 노드속성으로 부착 (event_vol 룰 입력)."""
    ret_path = OUTPUT_DIR / "sector_returns.parquet"
    if not ret_path.exists():
        print("  [WARN] sector_returns.parquet 없음, rv_zscore 스킵")
        return
    rets = pd.read_parquet(ret_path)
    count = 0
    for sec in SECTOR_NAMES:
        if sec not in rets.columns:
            continue
        z = compute_rv_zscore(rets[sec])
        if np.isfinite(z):
            nx.set_node_attributes(G, {sec: {"rv_zscore": round(float(z), 3)}})
            count += 1
    print(f"  [OK] rv_zscore 부착: {count}/{len(SECTOR_NAMES)} 섹터")


# ---------------------------------------------------------------------------
# Step 3: SENSITIVE_TO edges (delta/gamma from OLS + speed/color from poly4)
# ---------------------------------------------------------------------------

def _attach_fdr_q(G: nx.DiGraph) -> None:
    """SENSITIVE_TO 엣지 전체(family)에 BH FDR q값 부착 (2026-06-12).

    rate_* 룰의 유의성 게이트를 감사(trigger)와 동일한 q<0.10 기준으로 통일 —
    t>1.96 단독 게이트는 55개 가설 다중비교에서 우연 통과 2~3건을 허용한다.
    (trigger._bh_qvalues 와 동일 로직 — 순환 import 회피를 위한 의도적 중복.)
    """
    import math

    def _p(t):
        try:
            t = float(t)
        except (TypeError, ValueError):
            return float("nan")
        if not math.isfinite(t):
            return float("nan")
        return math.erfc(abs(t) / math.sqrt(2.0))

    edges = [(u, v) for u, v, d in G.edges(data=True)
             if d.get("edge_type") == "SENSITIVE_TO"]
    ps = [_p(G.edges[u, v].get("t_delta_ctrl")) for u, v in edges]
    finite = [i for i, p in enumerate(ps) if math.isfinite(p)]
    if not finite:
        return
    m = len(finite)
    order = sorted(finite, key=lambda i: ps[i])
    prev = 1.0
    qs = {i: float("nan") for i in range(len(ps))}
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        q = min(prev, ps[i] * m / rank)
        qs[i] = q
        prev = q
    for i, (u, v) in enumerate(edges):
        G.edges[u, v]["q_delta_ctrl"] = qs[i]


def _populate_sensitivity(G: nx.DiGraph) -> None:
    static  = _load_csv_safe(OUTPUT_DIR / "static_results.csv")
    partial = _load_csv_safe(OUTPUT_DIR / "partial_results.csv")
    poly4   = _load_csv_safe(OUTPUT_DIR / "poly4_static.csv")
    asymm   = _load_csv_safe(OUTPUT_DIR / "tail_asymmetry.csv")
    tdep_m  = _load_csv_safe(OUTPUT_DIR / "tail_dep_macro_lower.csv")
    if partial is None:
        print("  [WARN] partial_results.csv 없음 — rate_* 룰이 이변량(raw) delta 로 폴백")

    def _lookup_row(df: pd.DataFrame | None, sec: str, mac: str) -> pd.Series | None:
        """CSV가 (sector_index, macro_column) 구조이거나 compound_key 구조 양쪽을 지원."""
        if df is None:
            return None
        compound = f"{sec}_{mac}"
        if compound in df.index:
            return df.loc[compound]
        # (sector index, 'macro' column) 구조
        if "macro" in df.columns:
            mask = (df.index == sec) & (df["macro"] == mac)
            if mask.any():
                return df[mask].iloc[0]
        return None

    added = 0
    for sec in SECTOR_NAMES:
        for mac in MACRO_NAMES:
            edge = SensitiveToEdge()

            # OLS quadratic
            row = _lookup_row(static, sec, mac)
            if row is not None:
                edge.delta   = _safe(row.get("delta"))
                edge.gamma   = _safe(row.get("gamma"))
                edge.t_delta = _safe(row.get("t_delta"))
                edge.t_gamma = _safe(row.get("t_gamma"))
                edge.r2      = _safe(row.get("r2"))

            # 다변량 partial (전 매크로 동시 + SPY 통제) — rate_* 룰이 우선 사용
            row_p = _lookup_row(partial, sec, mac)
            if row_p is not None:
                edge.delta_ctrl   = _safe(row_p.get("delta_ctrl"))
                edge.gamma_ctrl   = _safe(row_p.get("gamma_ctrl"))
                edge.t_delta_ctrl = _safe(row_p.get("t_delta_ctrl"))
                edge.beta_mkt     = _safe(row_p.get("beta_mkt"))
                edge.vif          = _safe(row_p.get("vif"))

            # Poly4 (speed = 3rd order, color = 4th order)
            row4 = _lookup_row(poly4, sec, mac)
            if row4 is not None:
                edge.speed = _safe(row4.get("speed"))
                edge.color = _safe(row4.get("color"))

            # Tail asymmetry (quantile regression spread)
            row_asymm = _lookup_row(asymm, sec, mac)
            if row_asymm is not None:
                col = asymm.columns[0] if len(asymm.columns) == 1 else "tail_asymmetry"
                edge.tail_asymmetry = _safe(row_asymm.get(col, float("nan")))

            # Tail dependence sector -> macro
            if tdep_m is not None:
                if sec in tdep_m.index and mac in tdep_m.columns:
                    edge.tail_dep_lower = _safe(tdep_m.loc[sec, mac])

            G.add_edge(sec, mac, **edge.to_dict())
            added += 1

    _attach_fdr_q(G)
    print(f"  [OK] SENSITIVE_TO edges added: {added} (+ FDR q값 부착)")


# ---------------------------------------------------------------------------
# Step 4: CO_CRASH_WITH edges (lower tail dependence between sectors)
# ---------------------------------------------------------------------------

def _populate_co_crash(G: nx.DiGraph) -> None:
    lower = _load_csv_safe(OUTPUT_DIR / "tail_dep_sector_lower.csv")
    upper_path = OUTPUT_DIR / "tail_dep_sector_upper.csv"
    upper = pd.read_csv(upper_path, index_col=0) if upper_path.exists() else None
    # λ_L 부트스트랩 95% 상한 (2026-06-12) — thin_tail_greenlight 보수 게이트용
    hi_path = OUTPUT_DIR / "tail_dep_sector_lower_hi.csv"
    lower_hi = pd.read_csv(hi_path, index_col=0) if hi_path.exists() else None

    if lower is None:
        print("  [WARN] tail_dep_sector_lower.csv not found, skipping CO_CRASH_WITH")
        return

    sectors = list(SECTOR_NAMES.keys())
    added = 0
    for i, s1 in enumerate(sectors):
        for s2 in sectors[i + 1:]:
            lam_l = _safe(lower.loc[s1, s2]) if (s1 in lower.index and s2 in lower.columns) else float("nan")
            lam_u = float("nan")
            if upper is not None and s1 in upper.index and s2 in upper.columns:
                lam_u = _safe(upper.loc[s1, s2])
            lam_hi = float("nan")
            if lower_hi is not None and s1 in lower_hi.index and s2 in lower_hi.columns:
                lam_hi = _safe(lower_hi.loc[s1, s2])

            if not np.isnan(lam_l):
                edge = CoCrashEdge(lambda_lower=lam_l, lambda_upper=lam_u,
                                   lambda_lower_hi=lam_hi)
                G.add_edge(s1, s2, **edge.to_dict())
                added += 1

    print(f"  [OK] CO_CRASH_WITH edges added: {added}")


# ---------------------------------------------------------------------------
# Step 4b: Cholesky variance decomposition + CO_MOVES_WITH edges
# ---------------------------------------------------------------------------

# 어느 정도 기여돼야 엣지로 남길지 (자기 자신 제외)
_CO_MOVES_MIN_ATTR: float = 0.05  # 5% 이상이면 엣지 생성


def _populate_co_moves(G: nx.DiGraph) -> None:
    """sector_returns.parquet → Shapley-Owen 분산 분해 → 노드 var_decomposition
    + CO_MOVES_WITH.

    2026-06-10: Cholesky (centrality ordering, 순서 의존) → Shapley (ordering 독립,
    M=200 무작위 ordering 평균) 로 교체. 라운드 4 비평의 "centrality 1위 = 충격원" 라는
    임의 휴리스틱을 알고리즘 레벨에서 제거. 인과 의미는 여전히 없음 (Shapley 는 공정
    배분의 비인과 공식).

    실패 시 그래프 그대로 (다른 모듈에 영향 없음). 캐시가 없는 환경에서도 안전.
    """
    try:
        from ontology.mvg_mc import shapley_decompose
        d = shapley_decompose(n_samples=200)
    except FileNotFoundError as exc:
        print(f"  [WARN] shapley 스킵 — {exc}")
        return
    except Exception as exc:
        print(f"  [WARN] shapley 분해 실패: {exc}")
        return

    # 노드 속성 부착
    for i, t in enumerate(d.order):
        attr = {
            "var_decomposition": d.shock_contribution_to(t),
            "self_share":        d.self_share(t),
            "propagation_score": d.propagation_score(t),
            "cholesky_order":    i,
        }
        nx.set_node_attributes(G, {t: attr})

    # CO_MOVES_WITH 엣지: source(j) → target(i) when attribution >= threshold (i!=j)
    added = 0
    corr_df = d.corr
    for target in d.order:
        for source in d.order:
            if source == target:
                continue
            attr_val = float(d.var_attr.loc[target, source])
            if attr_val < _CO_MOVES_MIN_ATTR:
                continue
            pearson = float(corr_df.loc[target, source])
            edge = CoMovesEdge(attribution=attr_val, pearson_corr=pearson)
            # 같은 엣지(source→target)에 CO_CRASH_WITH 가 있을 수 있으나
            # 우리는 DiGraph의 동일 (u,v) 키 하나만 보유. CO_MOVES_WITH 가 더 정보량
            # 풍부하므로 덮어쓰지 않도록 분리 키로 저장.
            existing = G.get_edge_data(source, target)
            if existing and existing.get("edge_type") in (
                EdgeType.CO_CRASH_WITH.value, EdgeType.SENSITIVE_TO.value,
            ):
                # 이미 다른 엣지가 있는 (sec,sec) 쌍 — 노드 속성으로만 보존
                continue
            G.add_edge(source, target, **edge.to_dict())
            added += 1

    print(f"  [OK] CO_MOVES_WITH 엣지 추가: {added}개 "
          f"(self_share/propagation_score 노드속성 부착)")


# ---------------------------------------------------------------------------
# Step 5: Active regime (from latest VIX)
# ---------------------------------------------------------------------------

# 레짐 히스테리시스 (2026-06-12): VIX 가 15/25 경계를 넘나들 때 사이클마다 레짐이
# 플래핑하면 일시적 스파이크에 high_vix 룰이 발화해 포지션이 고착됨 (06-08 XLI 사례).
# 새 레짐은 연속 REGIME_CONFIRM_CYCLES 사이클 관측돼야 confirmed 로 전환.
# 상태는 output/regime_state.json 에 사이클 간 유지 (라이브 전용 — backtest_pipeline
# 은 _set_active_regime 를 쓰지 않고 자체 계산하므로 오염 없음).
REGIME_CONFIRM_CYCLES = 2
REGIME_STATE_JSON = OUTPUT_DIR / "regime_state.json"

_VALID_REGIMES = frozenset(
    r.value for r in (RegimeType.LOW_VIX, RegimeType.MID_VIX, RegimeType.HIGH_VIX))


def _confirm_regime(raw: str, vix: float) -> str:
    """사이클 간 히스테리시스. raw 레짐이 연속 N회 관측돼야 confirmed 전환."""
    import datetime as _dt
    import json as _json

    state: dict = {}
    if REGIME_STATE_JSON.exists():
        try:
            state = _json.loads(REGIME_STATE_JSON.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            state = {}

    confirmed = state.get("confirmed")
    pending = state.get("pending")
    count = int(state.get("pending_count", 0) or 0)

    if confirmed not in _VALID_REGIMES:
        confirmed, pending, count = raw, None, 0     # 첫 실행/상태 손상 → 즉시 채택
    elif raw == confirmed:
        pending, count = None, 0                      # 기존 레짐 유지 → 보류 리셋
    else:
        count = count + 1 if raw == pending else 1    # 새 레짐 관측 누적
        pending = raw
        if count >= REGIME_CONFIRM_CYCLES:
            confirmed, pending, count = raw, None, 0  # 연속 N회 확인 → 전환

    try:
        REGIME_STATE_JSON.write_text(_json.dumps({
            "confirmed": confirmed,
            "pending": pending,
            "pending_count": count,
            "vix": round(vix, 2),
            "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass   # 상태 저장 실패는 치명적이지 않음 — 다음 사이클에 재시도

    return confirmed


def _set_active_regime(G: nx.DiGraph) -> str:
    macro_path = OUTPUT_DIR / "macro_levels.parquet"
    if not macro_path.exists():
        print("  [WARN] macro_levels.parquet not found, defaulting to mid_vix")
        nx.set_node_attributes(G, {RegimeType.MID_VIX.value: {"active": True}})
        return RegimeType.MID_VIX.value

    macro_lvl = pd.read_parquet(macro_path)
    vix_latest = float(macro_lvl["VIX"].dropna().iloc[-1])

    if vix_latest < 15:
        raw_regime = RegimeType.LOW_VIX.value
    elif vix_latest < 25:
        raw_regime = RegimeType.MID_VIX.value
    else:
        raw_regime = RegimeType.HIGH_VIX.value

    regime = _confirm_regime(raw_regime, vix_latest)

    for r in [RegimeType.LOW_VIX.value, RegimeType.MID_VIX.value, RegimeType.HIGH_VIX.value]:
        nx.set_node_attributes(G, {r: {"active": r == regime, "vix_latest": vix_latest}})

    if regime != raw_regime:
        print(f"  [OK] Active regime: {regime}  (VIX={vix_latest:.2f}, raw={raw_regime} 보류 — "
              f"연속 {REGIME_CONFIRM_CYCLES}사이클 확인 대기)")
    else:
        print(f"  [OK] Active regime: {regime}  (VIX={vix_latest:.2f})")
    return regime


# ---------------------------------------------------------------------------
# Step 6: TRANSMITS_TO edges from PCMCI causal discovery
# ---------------------------------------------------------------------------

_CAUSAL_CSV = OUTPUT_DIR / "causal_links.csv"
_ALL_MACRO_NAMES = {**MACRO_NAMES, **CAUSAL_MACRO_NAMES}


def _populate_causal_links(G: nx.DiGraph) -> None:
    if not _CAUSAL_CSV.exists():
        print("  [WARN] causal_links.csv not found -- TRANSMITS_TO 엣지 스킵")
        return

    df = pd.read_csv(_CAUSAL_CSV)

    # historical 링크는 온톨로지에 포함하지 않음 (과거 레짐 잔재, 낮은 가중치)
    df = df[df["stability"] != "historical"].copy()

    # (source, target) 쌍별로 가장 높은 score 행만 유지
    # (같은 pair에 여러 lag가 있을 때 가장 유의미한 것 선택)
    best = (df.sort_values("score", ascending=False)
              .groupby(["source", "target", "direction"], as_index=False)
              .first())

    added = 0
    for _, row in best.iterrows():
        src = str(row["source"])
        dst = str(row["target"])

        # 노드가 없으면 동적으로 추가 (PCMCI 변수가 MACRO_NAMES에 없을 수 있음)
        for nid in (src, dst):
            if nid not in G.nodes:
                from ontology.schema import MacroFactorNode
                node = MacroFactorNode(ticker=nid, name=_ALL_MACRO_NAMES.get(nid, nid))
                G.add_node(nid, **node.to_dict())

        # coef: structural은 full 기간, emerging은 recent 기간 사용
        if row["stability"] == "structural":
            coef = _safe(row.get("coef_full", float("nan")))
            pval = _safe(row.get("p_full", float("nan")))
        else:
            coef = _safe(row.get("coef_recent", float("nan")))
            pval = _safe(row.get("p_recent", float("nan")))

        edge = TransmitsToEdge(
            lag_months = int(row["lag_months"]),
            coef       = coef,
            p_value    = pval,
            stability  = str(row["stability"]),
            score      = _safe(row["score"]),
            direction  = str(row["direction"]),
        )

        # 같은 (src, dst)에 기존 엣지가 있으면 score가 높은 것만 유지
        existing = G.get_edge_data(src, dst)
        if existing and existing.get("edge_type") == EdgeType.TRANSMITS_TO.value:
            if _safe(existing.get("score", 0)) >= edge.score:
                continue

        G.add_edge(src, dst, **edge.to_dict())
        added += 1

    print(f"  [OK] TRANSMITS_TO 엣지 추가: {added}개 "
          f"(structural+emerging, best-score per pair)")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def populate(G: nx.DiGraph) -> str:
    """Populate G in-place. Returns the active regime string."""
    print("[populate] Loading tail risk ...")
    _populate_tail_risk(G)

    print("[populate] Loading VRP ...")
    _populate_vrp(G)

    print("[populate] Extracting ATM IV (Alpaca options) → VRP_true ...")
    _populate_vrp_iv(G)

    print("[populate] Realized-vol z-score (event_vol 입력) ...")
    _populate_rv_zscore(G)

    print("[populate] Building SENSITIVE_TO edges ...")
    _populate_sensitivity(G)

    print("[populate] Building CO_CRASH_WITH edges ...")
    _populate_co_crash(G)

    print("[populate] Cholesky 분산 분해 + CO_MOVES_WITH 엣지 ...")
    _populate_co_moves(G)

    print("[populate] Setting active regime ...")
    active_regime = _set_active_regime(G)

    print("[populate] Building TRANSMITS_TO causal links ...")
    _populate_causal_links(G)

    print(f"[populate] Done. Nodes={G.number_of_nodes()}  Edges={G.number_of_edges()}")
    return active_regime


if __name__ == "__main__":
    from ontology.schema import build_empty_graph
    G = build_empty_graph()
    regime = populate(G)
    print(f"\nActive regime: {regime}")
    print("Sector node sample (XLK):", dict(G.nodes["XLK"]))
    edge_data = G.get_edge_data("XLK", "VIX")
    print("XLK->VIX edge:", edge_data)

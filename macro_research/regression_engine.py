"""
regression_engine.py — 섹터 수익률 × 매크로 변수 2차(quadratic) 회귀

모형:
    R_s = α + δ·ΔX + γ·ΔX² + ε

    δ (delta) : 선형 민감도  — ΔX 1단위 변화에 대한 섹터 수익률 반응
    γ (gamma) : 비선형 볼록성 — δ 자체가 ΔX에 따라 어떻게 변하는지
                γ > 0 → 볼록(convex): 극단 충격에서 상대적으로 덜 손상
                γ < 0 → 오목(concave): 극단 충격에서 가속 손상

회귀 전 ΔX를 표준화(z-score)하여 4개 매크로 변수 간 delta/gamma 크기를 비교 가능하게 한다.
원래 단위 계수는 별도로 보관한다.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import SECTOR_ETFS, MACRO_TICKERS, ROLLING_WINDOW


# ─────────────────────────────────────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

_RESULT_KEYS = ["alpha", "delta", "gamma", "t_alpha", "t_delta", "t_gamma",
                "r2", "r2_adj", "n"]


def _ols_quadratic(y: np.ndarray, x: np.ndarray) -> dict:
    """
    단일 구간 2차 OLS. y와 x에서 NaN을 제거 후 최소 30개 이상 관측치가 있어야 실행.
    반환: alpha, delta, gamma, t-stats, R², n
    """
    mask = ~(np.isnan(y) | np.isnan(x))
    y, x = y[mask], x[mask]

    if len(y) < 30:
        return {k: np.nan for k in _RESULT_KEYS}

    X = sm.add_constant(np.column_stack([x, x ** 2]), has_constant="add")
    try:
        res = sm.OLS(y, X).fit()
        return {
            "alpha":   res.params[0],  "t_alpha":  res.tvalues[0],
            "delta":   res.params[1],  "t_delta":  res.tvalues[1],
            "gamma":   res.params[2],  "t_gamma":  res.tvalues[2],
            "r2":      res.rsquared,
            "r2_adj":  res.rsquared_adj,
            "n":       int(res.nobs),
        }
    except Exception:
        return {k: np.nan for k in _RESULT_KEYS}


def _standardize(series: pd.Series) -> pd.Series:
    """전체 기간 μ, σ로 z-score 정규화."""
    mu, sigma = series.mean(), series.std()
    if sigma == 0 or np.isnan(sigma):
        return series
    return (series - mu) / sigma


# ─────────────────────────────────────────────────────────────────────────────
# 전체 기간 정적 회귀
# ─────────────────────────────────────────────────────────────────────────────

def run_static_regression(
    returns: pd.DataFrame,
    macro_changes: pd.DataFrame,
    standardize: bool = True,
) -> pd.DataFrame:
    """
    모든 (sector, macro) 조합에 대해 전체 기간 2차 OLS를 실행한다.

    standardize=True 이면 ΔX를 z-score 변환하여 계수 크기를 비교 가능하게 한다.

    반환: MultiIndex(sector, macro) × [alpha, delta, gamma, t_*, r2, r2_adj, n]
    """
    sectors = [c for c in returns.columns if c != "SPY"]
    macros  = [c for c in macro_changes.columns if c in MACRO_TICKERS]

    common = returns.index.intersection(macro_changes.index)
    R = returns.loc[common]
    M = macro_changes.loc[common]

    if standardize:
        M = M.apply(_standardize)

    rows = []
    for sector in sectors:
        for macro in macros:
            stats = _ols_quadratic(R[sector].values, M[macro].values)
            rows.append({"sector": sector, "macro": macro, **stats})

    return pd.DataFrame(rows).set_index(["sector", "macro"])


# ─────────────────────────────────────────────────────────────────────────────
# 롤링 회귀
# ─────────────────────────────────────────────────────────────────────────────

def run_rolling_regression(
    returns: pd.DataFrame,
    macro_changes: pd.DataFrame,
    window: int = ROLLING_WINDOW,
    standardize: bool = True,
) -> dict[tuple[str, str], pd.DataFrame]:
    """
    63일 롤링 2차 OLS.

    반환: dict 키=(sector, macro), 값=DataFrame(date index, [alpha,delta,gamma,t_*,r2,n])
    """
    sectors = [c for c in returns.columns if c != "SPY"]
    macros  = [c for c in macro_changes.columns if c in MACRO_TICKERS]

    common = returns.index.intersection(macro_changes.index)
    R = returns.loc[common]
    M = macro_changes.loc[common].copy()

    if standardize:
        # 롤링 내에서가 아닌 전체 μ/σ로 정규화 (look-ahead 없음: 전체 기간 σ 사용)
        M = M.apply(_standardize)

    results: dict[tuple[str, str], pd.DataFrame] = {}
    n_total = len(sectors) * len(macros)
    counter = 0

    for sector in sectors:
        for macro in macros:
            counter += 1
            print(f"  Rolling [{counter}/{n_total}] {sector} × {macro}...")

            y_arr = R[sector].values
            x_arr = M[macro].values
            dates  = R.index

            rows = []
            for end in range(window, len(y_arr) + 1):
                start = end - window
                stats = _ols_quadratic(y_arr[start:end], x_arr[start:end])
                rows.append({"date": dates[end - 1], **stats})

            results[(sector, macro)] = pd.DataFrame(rows).set_index("date")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 다변량(부분) 회귀 — 교란 통제 (2026-06-12)
# ─────────────────────────────────────────────────────────────────────────────

_MV_COLS = ["delta_ctrl", "t_delta_ctrl", "p_delta_ctrl", "gamma_ctrl",
            "t_gamma_ctrl", "beta_mkt", "t_mkt", "vif", "r2", "r2_adj", "n"]


def run_multivariate_regression(
    returns: pd.DataFrame,
    macro_changes: pd.DataFrame,
    standardize: bool = True,
    market_col: str = "SPY",
) -> pd.DataFrame:
    """
    모든 매크로를 *동시 투입* + 시장수익률(SPY) 통제 다변량 2차 OLS.

        R_s = α + Σ_j δ_j·ΔX_j + Σ_j γ_j·ΔX_j² + β_mkt·R_SPY + ε

    이변량(run_static_regression)의 delta 는 누락변수 교란에 노출된다:
    예) 유가↑ → 금리↑ 가 같은 표본에서 동시 발생하면, OIL 을 빼고 추정한
    delta(US10Y) 가 유가 효과를 흡수해 XLE 가 "금리 수혜" 로 보인다.
    SPY 통제는 시장 베타 교란 제거 — Fama-French MKT 팩터 통제와 같은 역할.

    2026-06-12 개정 (라운드 6 비평 7번 — 공선성 vs 교란 구분):
      ΔUS10Y·ΔUS2Y 동시 투입은 상관이 높아 VIF 가 커지고 t 가 죽는다 —
      그러면 'KILLED' 가 교란 제거인지 분산 팽창인지 구분 불가. 직교화:
        US10Y  = ΔUS10Y (레벨)
        T10Y2Y = Δ(US10Y − US2Y) (2s10s 기울기)
      US2Y 효과는 contrast (레벨 − 기울기) 로 복원 (t 는 t_test, gamma 는 NaN).
      각 행에 VIF 기록 — 공선성 진단 가능. p_delta_ctrl 로 FDR 보정 지원.

    δ_j 해석: 다른 매크로와 시장을 고정한 상태에서 ΔX_j 1σ 변화의 한계 효과.
    반환: MultiIndex(sector, macro) × _MV_COLS
    """
    sectors = [c for c in returns.columns if c != market_col]
    base = [c for c in macro_changes.columns if c in MACRO_TICKERS]

    common = returns.index.intersection(macro_changes.index)
    R = returns.loc[common]
    M = macro_changes.loc[common]

    # ── 직교화 기저 구성 ──
    design: dict[str, pd.Series] = {}
    for m in base:
        if m in ("US10Y", "US2Y"):
            continue
        design[m] = M[m]
    has_both_rates = "US10Y" in M.columns and "US2Y" in M.columns
    if has_both_rates:
        design["US10Y"]  = M["US10Y"]                # 레벨
        design["T10Y2Y"] = M["US10Y"] - M["US2Y"]    # 기울기 (2s10s) 변화
    elif "US10Y" in M.columns:
        design["US10Y"] = M["US10Y"]
    elif "US2Y" in M.columns:
        design["US2Y"] = M["US2Y"]

    D = pd.DataFrame(design)
    if standardize:
        D = D.apply(_standardize)
    reg_names = list(D.columns)
    k = len(reg_names)

    mkt = R[market_col] if market_col in R.columns else None

    # ── VIF (선형부 공통 — 섹터 무관) ──
    vif_map: dict[str, float] = {}
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        lin_cols = [D[m] for m in reg_names]
        if mkt is not None:
            lin_cols.append(mkt.rename(market_col))
        L = pd.concat(lin_cols, axis=1).dropna()
        Lc = sm.add_constant(L.values, has_constant="add")
        for j, name in enumerate(list(L.columns)):
            vif_map[name] = float(variance_inflation_factor(Lc, j + 1))
    except Exception:
        pass

    rows = []
    for sector in sectors:
        cols = [D[m] for m in reg_names] + [D[m] ** 2 for m in reg_names]
        if mkt is not None:
            cols.append(mkt)
        X_df = pd.concat(cols, axis=1)
        y = R[sector]
        mask = ~(X_df.isna().any(axis=1) | y.isna())
        Xv, yv = X_df[mask].values, y[mask].values

        min_obs = 30 + 2 * k
        res = None
        if len(yv) >= min_obs:
            Xc = sm.add_constant(Xv, has_constant="add")
            try:
                res = sm.OLS(yv, Xc).fit()
            except Exception:
                res = None

        def _nan_row(macro: str) -> dict:
            return {"sector": sector, "macro": macro,
                    **{c: np.nan for c in _MV_COLS}}

        if res is None:
            for m in reg_names:
                rows.append(_nan_row(m))
            if has_both_rates:
                rows.append(_nan_row("US2Y"))
            continue

        for j, m in enumerate(reg_names):
            rows.append({
                "sector": sector, "macro": m,
                "delta_ctrl":   res.params[1 + j],
                "t_delta_ctrl": res.tvalues[1 + j],
                "p_delta_ctrl": res.pvalues[1 + j],
                "gamma_ctrl":   res.params[1 + k + j],
                "t_gamma_ctrl": res.tvalues[1 + k + j],
                "beta_mkt":     res.params[1 + 2 * k] if mkt is not None else np.nan,
                "t_mkt":        res.tvalues[1 + 2 * k] if mkt is not None else np.nan,
                "vif":          vif_map.get(m, np.nan),
                "r2":           res.rsquared,
                "r2_adj":       res.rsquared_adj,
                "n":            int(res.nobs),
            })

        # US2Y 복원: ΔUS2Y = ΔUS10Y − Δ(10Y−2Y) → 계수 = b_lvl − b_slope (contrast)
        if has_both_rates:
            try:
                n_par = len(res.params)
                Lvec = np.zeros(n_par)
                Lvec[1 + reg_names.index("US10Y")]  = 1.0
                Lvec[1 + reg_names.index("T10Y2Y")] = -1.0
                ct = res.t_test(Lvec)
                rows.append({
                    "sector": sector, "macro": "US2Y",
                    "delta_ctrl":   float(np.atleast_1d(ct.effect)[0]),
                    "t_delta_ctrl": float(np.atleast_1d(ct.tvalue)[0]),
                    "p_delta_ctrl": float(np.atleast_1d(ct.pvalue)[0]),
                    "gamma_ctrl":   np.nan,   # 차분 제곱은 선형결합 아님
                    "t_gamma_ctrl": np.nan,
                    "beta_mkt":     res.params[1 + 2 * k] if mkt is not None else np.nan,
                    "t_mkt":        res.tvalues[1 + 2 * k] if mkt is not None else np.nan,
                    "vif":          np.nan,
                    "r2":           res.rsquared,
                    "r2_adj":       res.rsquared_adj,
                    "n":            int(res.nobs),
                })
            except Exception:
                rows.append(_nan_row("US2Y"))

    return pd.DataFrame(rows).set_index(["sector", "macro"])


# ─────────────────────────────────────────────────────────────────────────────
# 요약 피벗
# ─────────────────────────────────────────────────────────────────────────────

def pivot_summary(
    static_df: pd.DataFrame,
    coef: str = "gamma",
    t_threshold: float = 1.96,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    coef 피벗, t-stat 피벗, 유의성 마스크(bool)를 반환.

    t_threshold=1.96 → 5% 양측 유의수준
    """
    t_col    = f"t_{coef}"
    val_piv  = static_df[coef].unstack("macro")
    t_piv    = static_df[t_col].unstack("macro")
    sig_piv  = t_piv.abs() > t_threshold
    return val_piv, t_piv, sig_piv


# ─────────────────────────────────────────────────────────────────────────────
# 단독 실행: 캐시 parquet → static + partial 결과 갱신
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    from config import OUTPUT_DIR

    returns   = pd.read_parquet(OUTPUT_DIR / "sector_returns.parquet")
    macro_chg = pd.read_parquet(OUTPUT_DIR / "macro_changes.parquet")

    static_df = run_static_regression(returns, macro_chg, standardize=True)
    static_df.to_csv(OUTPUT_DIR / "static_results.csv")
    print(f"static_results.csv  갱신 ({len(static_df)} rows)")

    partial_df = run_multivariate_regression(returns, macro_chg, standardize=True)
    partial_df.to_csv(OUTPUT_DIR / "partial_results.csv")
    print(f"partial_results.csv 갱신 ({len(partial_df)} rows)")

    # 이변량 vs 통제 delta 비교 표 (교란 진단)
    cmp = pd.DataFrame({
        "delta_raw":  static_df["delta"],
        "t_raw":      static_df["t_delta"],
        "delta_ctrl": partial_df["delta_ctrl"],
        "t_ctrl":     partial_df["t_delta_ctrl"],
    }).dropna(how="all")
    print("\n=== 이변량(raw) vs 통제(ctrl) delta — |t_raw|>1.96 만 ===")
    sig = cmp[cmp["t_raw"].abs() > 1.96].sort_values("t_raw", key=abs, ascending=False)
    print(sig.round(4).to_string())

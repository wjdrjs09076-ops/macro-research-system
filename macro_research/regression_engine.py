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

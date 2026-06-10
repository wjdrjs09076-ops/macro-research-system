"""
higher_order_regression.py - 4th order polynomial regression + kurtosis analysis

Model (orthogonal polynomial basis — R's poly() equivalent):
    R = a + c1*P1(X) + c2*P2(X) + c3*P3(X) + c4*P4(X) + e

    Orthogonalization via QR decomposition eliminates multicollinearity.
    Each Pi(X) is uncorrelated with Pj(X), so each coefficient is estimated
    independently regardless of the others.

    c1 ~ delta  (linear sensitivity)
    c2 ~ gamma  (convexity / quadratic)
    c3 ~ speed  (rate of change of gamma / cubic)
    c4 ~ color  (kurtosis sensitivity / quartic)

Kurtosis analysis:
    - Excess kurtosis of sector returns (full period)
    - Rolling 126D kurtosis time series
    - Conditional kurtosis by VIX regime (low / mid / high)
    - Kurtosis of regression residuals (unexplained by macro factors)
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
import statsmodels.api as sm

from config import OUTPUT_DIR, FIGURES_DIR, ROLLING_DIR, SECTOR_ETFS, MACRO_TICKERS


# ---------------------------------------------------------------------------
# Orthogonal polynomial basis  (equivalent to R's poly())
# ---------------------------------------------------------------------------

def orthopoly_basis(x: np.ndarray, degree: int = 4) -> np.ndarray:
    """
    Build orthogonal polynomial basis up to given degree via QR decomposition.

    Input : x shape (n,) — may contain NaN (excluded from QR, filled back as NaN)
    Output: matrix shape (n, degree) — columns are mutually orthogonal
            (intercept NOT included; add with sm.add_constant separately)
    """
    mask = ~np.isnan(x)
    x_clean = x[mask]
    n = len(x_clean)

    # Vandermonde matrix: [x^0, x^1, ..., x^degree]
    V = np.column_stack([x_clean ** i for i in range(degree + 1)])

    # QR decomposition — Q columns are orthogonal
    Q, _ = np.linalg.qr(V)

    # Scale so each column has unit variance (for coefficient comparability)
    Q_scaled = Q * np.sqrt(n)

    # Drop the first column (corresponds to constant / x^0)
    Q_poly = Q_scaled[:, 1:]  # shape (n, degree)

    # Re-insert NaN rows
    result = np.full((len(x), degree), np.nan)
    result[mask] = Q_poly
    return result


# ---------------------------------------------------------------------------
# Single regression fit
# ---------------------------------------------------------------------------

_ORDERS = ["delta", "gamma", "speed", "color"]

def fit_poly4_ols(y: np.ndarray, x: np.ndarray) -> dict:
    """
    4th degree orthogonal polynomial OLS.
    Returns dict: {alpha, delta, gamma, speed, color, t_*, r2, r2_adj, n}
    """
    basis = orthopoly_basis(x, degree=4)  # (n, 4)
    mask = ~(np.isnan(y) | np.any(np.isnan(basis), axis=1))
    y_c, B_c = y[mask], basis[mask]

    if len(y_c) < 30:
        keys = ["alpha"] + _ORDERS + [f"t_{k}" for k in ["alpha"] + _ORDERS] + ["r2", "r2_adj", "n"]
        return {k: np.nan for k in keys}

    X_const = sm.add_constant(B_c, has_constant="add")
    try:
        res = sm.OLS(y_c, X_const).fit()
    except Exception:
        keys = ["alpha"] + _ORDERS + [f"t_{k}" for k in ["alpha"] + _ORDERS] + ["r2", "r2_adj", "n"]
        return {k: np.nan for k in keys}

    names = ["alpha"] + _ORDERS
    out = {}
    for i, name in enumerate(names):
        out[name]        = res.params[i]
        out[f"t_{name}"] = res.tvalues[i]
    out["r2"]      = res.rsquared
    out["r2_adj"]  = res.rsquared_adj
    out["n"]       = int(res.nobs)
    return out


# ---------------------------------------------------------------------------
# Static regression (full period)
# ---------------------------------------------------------------------------

def run_static_poly4(
    returns: pd.DataFrame,
    macro_changes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Full-period 4th order regression for each (sector, macro) pair.
    ΔX standardized before polynomial expansion.
    Returns MultiIndex(sector, macro) DataFrame.
    """
    sectors = [c for c in returns.columns if c != "SPY"]
    macros  = [c for c in macro_changes.columns if c in MACRO_TICKERS]

    common = returns.index.intersection(macro_changes.index)
    R = returns.loc[common]
    M = macro_changes.loc[common].apply(lambda s: (s - s.mean()) / (s.std() or 1))

    rows = []
    for sector in sectors:
        for macro in macros:
            stats = fit_poly4_ols(R[sector].values, M[macro].values)
            rows.append({"sector": sector, "macro": macro, **stats})

    return pd.DataFrame(rows).set_index(["sector", "macro"])


# ---------------------------------------------------------------------------
# Rolling regression (126D window)
# ---------------------------------------------------------------------------

def run_rolling_poly4(
    returns: pd.DataFrame,
    macro_changes: pd.DataFrame,
    window: int = 126,
) -> dict[tuple[str, str], pd.DataFrame]:
    """
    Rolling 4th order regression. Window=126D (recommended for 17-param model).
    Returns dict: (sector, macro) -> DataFrame(date, alpha, delta, gamma, speed, color, t_*, r2)
    """
    sectors = [c for c in returns.columns if c != "SPY"]
    macros  = [c for c in macro_changes.columns if c in MACRO_TICKERS]

    common = returns.index.intersection(macro_changes.index)
    R = returns.loc[common]
    M = macro_changes.loc[common].apply(lambda s: (s - s.mean()) / (s.std() or 1))

    results = {}
    total   = len(sectors) * len(macros)
    counter = 0

    y_arrs = {sec: R[sec].values          for sec in sectors}
    x_arrs = {mac: M[mac].values          for mac in macros}
    dates  = R.index

    for sector in sectors:
        for macro in macros:
            counter += 1
            print(f"  [{counter:>2}/{total}] {sector} x {macro}...", flush=True)

            y_arr = y_arrs[sector]
            x_arr = x_arrs[macro]

            rows = []
            for end in range(window, len(y_arr) + 1):
                s = end - window
                stats = fit_poly4_ols(y_arr[s:end], x_arr[s:end])
                rows.append({"date": dates[end - 1], **stats})

            results[(sector, macro)] = pd.DataFrame(rows).set_index("date")

    return results


# ---------------------------------------------------------------------------
# Kurtosis analysis
# ---------------------------------------------------------------------------

def compute_kurtosis_stats(
    returns: pd.DataFrame,
    macro_levels: pd.DataFrame,
    window: int = 126,
) -> dict:
    """
    Returns:
        full_kurtosis   : excess kurtosis per sector (full period)
        rolling_kurtosis: rolling 126D excess kurtosis DataFrame
        cond_kurtosis   : kurtosis by VIX regime {low/mid/high} per sector
    """
    sectors = [c for c in returns.columns if c != "SPY"]

    # Full-period excess kurtosis
    full_kurt = returns[sectors].apply(ss.kurtosis)  # Fisher: normal=0

    # Rolling kurtosis
    rolling_kurt = returns[sectors].rolling(window, min_periods=window // 2).kurt()

    # Conditional kurtosis by VIX regime
    vix = macro_levels["VIX"].reindex(returns.index).ffill()
    low_mask  = vix < 15
    mid_mask  = (vix >= 15) & (vix < 25)
    high_mask = vix >= 25

    cond_kurt = {}
    for regime, mask in [("low_vix (<15)", low_mask),
                         ("mid_vix (15-25)", mid_mask),
                         ("high_vix (>25)", high_mask)]:
        subset = returns[sectors].loc[mask]
        cond_kurt[regime] = subset.apply(ss.kurtosis)

    cond_df = pd.DataFrame(cond_kurt)

    return {
        "full":    full_kurt,
        "rolling": rolling_kurt,
        "conditional": cond_df,
    }


def compute_residual_kurtosis(
    returns: pd.DataFrame,
    macro_changes: pd.DataFrame,
) -> pd.Series:
    """
    Kurtosis of OLS residuals after removing macro factor (4th order) effects.
    Residual kurtosis >> full kurtosis -> macro factors explain fat tails.
    Residual kurtosis ~= full kurtosis -> idiosyncratic fat tails dominate.
    """
    sectors = [c for c in returns.columns if c != "SPY"]
    macros  = [c for c in macro_changes.columns if c in MACRO_TICKERS]

    common = returns.index.intersection(macro_changes.index)
    R = returns.loc[common]
    M = macro_changes.loc[common].apply(lambda s: (s - s.mean()) / (s.std() or 1))

    residual_kurt = {}
    for sector in sectors:
        y = R[sector].dropna()
        # Stack all macro polynomial bases
        basis_cols = []
        for mac in macros:
            x = M[mac].reindex(y.index).values
            basis_cols.append(orthopoly_basis(x, degree=4))

        B = np.hstack(basis_cols)  # (n, 16)
        mask = ~(np.isnan(y.values) | np.any(np.isnan(B), axis=1))
        y_c, B_c = y.values[mask], B[mask]

        if len(y_c) < 50:
            residual_kurt[sector] = np.nan
            continue

        X_const = sm.add_constant(B_c, has_constant="add")
        try:
            res = sm.OLS(y_c, X_const).fit()
            residual_kurt[sector] = float(ss.kurtosis(res.resid))
        except Exception:
            residual_kurt[sector] = np.nan

    return pd.Series(residual_kurt, name="residual_kurtosis")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_poly4_heatmap(
    static_df: pd.DataFrame,
    order: str,
    filename: str,
):
    """Heatmap for a single polynomial order (speed or color)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    t_col   = f"t_{order}"
    val_piv = static_df[order].unstack("macro")
    t_piv   = static_df[t_col].unstack("macro")
    sig_piv = t_piv.abs() > 1.96

    def _star(t):
        a = abs(t)
        if np.isnan(a): return ""
        if a > 2.576:   return "***"
        if a > 1.96:    return "**"
        if a > 1.645:   return "*"
        return ""

    annot = val_piv.copy().astype(object)
    for r in val_piv.index:
        for c in val_piv.columns:
            v = val_piv.loc[r, c]
            t = t_piv.loc[r, c]
            annot.loc[r, c] = "N/A" if np.isnan(v) else f"{v:.5f}\n{_star(t)}"

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(val_piv, ax=ax, cmap="PuOr", center=0,
                annot=annot, fmt="", linewidths=0.5,
                cbar_kws={"label": f"{order} coefficient"},
                mask=~sig_piv)
    grey = pd.DataFrame(
        np.where(~sig_piv.values, val_piv.values, np.nan),
        index=val_piv.index, columns=val_piv.columns)
    sns.heatmap(grey, ax=ax, cmap="Greys", center=0,
                vmin=val_piv.min().min(), vmax=val_piv.max().max(),
                annot=annot, fmt="", linewidths=0.5, cbar=False,
                alpha=0.35, mask=sig_piv)

    order_labels = {"speed": "3rd order (Speed)", "color": "4th order (Color/Kurtosis sensitivity)"}
    ax.set_title(
        f"Sector {order_labels.get(order, order)} to Macro Variables (full period)\n"
        "[colored = |t|>1.96  |  ** p<5%  *** p<1%]",
        fontsize=11, pad=10,
    )
    ax.set_xlabel("Macro Variable")
    ax.set_ylabel("Sector ETF")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def plot_rolling_order(
    rolling_results: dict,
    sector: str,
    macros: list[str],
    order: str,
    filename: str,
):
    """Rolling time series for one polynomial order across all macros for one sector."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    n = len(macros)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.8 * n))
    if n == 1:
        axes = [axes]

    colors = {"speed": "purple", "color": "firebrick"}
    color  = colors.get(order, "steelblue")

    for i, macro in enumerate(macros):
        key = (sector, macro)
        ax  = axes[i]
        if key not in rolling_results:
            continue
        df = rolling_results[key]
        df[order].plot(ax=ax, color=color, linewidth=0.9)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.fill_between(df.index, df[order], 0,
                        where=df[order] > 0, alpha=0.12, color="green")
        ax.fill_between(df.index, df[order], 0,
                        where=df[order] <= 0, alpha=0.12, color="red")
        ax.set_title(f"{sector} x {macro} -- {order} (126D rolling)", fontsize=9)
        ax.tick_params(labelsize=7)

    plt.suptitle(f"{order.capitalize()} rolling: {sector}", fontsize=10, y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=130, bbox_inches="tight")
    plt.close()


def plot_kurtosis(kurt_stats: dict, filename: str = "kurtosis_analysis.png"):
    """Kurtosis bar charts: full period, by VIX regime, conditional."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    full  = kurt_stats["full"]
    cond  = kurt_stats["conditional"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Full period kurtosis
    ax1 = axes[0]
    colors = ["tomato" if v > 0 else "steelblue" for v in full.values]
    ax1.bar(full.index, full.values, color=colors, edgecolor="black", linewidth=0.5)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.axhline(3, color="grey", linewidth=0.7, linestyle="--", label="Normal dist (+3)")
    ax1.set_title("Excess Kurtosis by Sector (full period 2014-2026)\n"
                  ">0 = fat tails (leptokurtic)  <0 = thin tails", fontsize=10)
    ax1.set_ylabel("Excess Kurtosis (Fisher, normal=0)")
    ax1.tick_params(axis="x", rotation=45, labelsize=8)
    ax1.legend(fontsize=8)

    # Conditional kurtosis by VIX regime
    ax2 = axes[1]
    x = np.arange(len(full.index))
    w = 0.25
    regime_colors = ["#2196F3", "#FF9800", "#F44336"]
    for i, (regime, row) in enumerate(cond.items()):
        vals = [row.get(sec, np.nan) for sec in full.index]
        ax2.bar(x + i * w, vals, w, label=regime,
                color=regime_colors[i], alpha=0.8, edgecolor="black", linewidth=0.4)
    ax2.axhline(0, color="black", linewidth=0.7)
    ax2.set_xticks(x + w)
    ax2.set_xticklabels(full.index, rotation=45, fontsize=8)
    ax2.set_title("Conditional Excess Kurtosis by VIX Regime\n"
                  "High VIX regime -> fatter tails?", fontsize=10)
    ax2.set_ylabel("Excess Kurtosis")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def plot_rolling_kurtosis(rolling_kurt: pd.DataFrame, filename: str = "rolling_kurtosis.png"):
    """Rolling 126D excess kurtosis per sector."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    for col in rolling_kurt.columns:
        rolling_kurt[col].rolling(21).mean().plot(
            ax=ax, linewidth=0.9, label=col, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_title("Rolling Excess Kurtosis (126D window, 21D smoothed)\n"
                 "Spikes = periods of fat-tailed returns", fontsize=10)
    ax.set_ylabel("Excess Kurtosis")
    ax.legend(loc="upper right", fontsize=7, ncol=4)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ROLLING_DIR.mkdir(parents=True, exist_ok=True)

    sep = "=" * 62

    # Load cached data
    returns   = pd.read_parquet(OUTPUT_DIR / "sector_returns.parquet")
    macro_chg = pd.read_parquet(OUTPUT_DIR / "macro_changes.parquet")
    macro_lvl = pd.read_parquet(OUTPUT_DIR / "macro_levels.parquet")

    sectors = [c for c in returns.columns if c != "SPY"]
    macros  = list(macro_chg.columns)

    # STEP 1: Static 4th order regression -----------------------------------
    print(sep)
    print("STEP 1 - Static 4th order polynomial regression (full period)")
    print(sep)

    static4 = run_static_poly4(returns, macro_chg)
    static4.to_csv(OUTPUT_DIR / "poly4_static.csv")

    for order in _ORDERS:
        piv = static4[order].unstack("macro")
        t   = static4[f"t_{order}"].unstack("macro")
        sig = (t.abs() > 1.96).sum().sum()
        print(f"\n  {order.upper()} - {sig}/{len(piv)*len(piv.columns)} significant (|t|>1.96)")
        print(piv.round(5).to_string())

    # Heatmaps for speed and color (delta/gamma already done in run_analysis.py)
    plot_poly4_heatmap(static4, "speed", "heatmap_speed.png")
    plot_poly4_heatmap(static4, "color", "heatmap_color.png")

    # Also save delta/gamma from poly4 for comparison
    print("\n  R2_adj comparison (2nd order vs 4th order):")
    r2_poly4 = static4["r2_adj"].unstack("macro")
    print(r2_poly4.round(4).to_string())

    # STEP 2: Rolling 4th order regression (126D) ---------------------------
    print(f"\n{sep}")
    print("STEP 2 - Rolling 4th order regression (126D window)")
    print(sep)

    rolling4 = run_rolling_poly4(returns, macro_chg, window=126)

    for (sector, macro), df in rolling4.items():
        df.to_csv(ROLLING_DIR / f"poly4_{sector}_{macro}.csv")

    # Rolling charts for speed and color (select key sectors)
    key_sectors = ["XLK", "XLF", "XLE", "XLU", "XLRE"]
    for sec in key_sectors:
        for order in ["speed", "color"]:
            plot_rolling_order(rolling4, sec, macros, order,
                               f"rolling_{order}_{sec}.png")

    print(f"\n  Rolling charts saved -> {FIGURES_DIR}")

    # STEP 3: Kurtosis analysis ---------------------------------------------
    print(f"\n{sep}")
    print("STEP 3 - Kurtosis analysis")
    print(sep)

    kurt_stats = compute_kurtosis_stats(returns, macro_lvl, window=126)

    print("\n  Full-period excess kurtosis (normal=0, fat tails > 0):")
    full_k = kurt_stats["full"].sort_values(ascending=False)
    for sec, k in full_k.items():
        bar = "#" * int(max(0, k) * 5)
        print(f"  {sec:5s}: {k:+.3f}  {bar}")

    print("\n  Conditional kurtosis by VIX regime:")
    print(kurt_stats["conditional"].round(3).to_string())

    print("\n  Residual kurtosis (after removing all macro poly4 factors):")
    resid_kurt = compute_residual_kurtosis(returns, macro_chg)
    compare = pd.DataFrame({
        "full_kurtosis":     kurt_stats["full"],
        "residual_kurtosis": resid_kurt,
        "reduction":         kurt_stats["full"] - resid_kurt,
    }).sort_values("reduction", ascending=False)
    print(compare.round(3).to_string())
    compare.to_csv(OUTPUT_DIR / "kurtosis_comparison.csv")

    kurt_stats["full"].to_csv(OUTPUT_DIR / "kurtosis_full.csv")
    kurt_stats["rolling"].to_csv(OUTPUT_DIR / "kurtosis_rolling.csv")
    kurt_stats["conditional"].to_csv(OUTPUT_DIR / "kurtosis_conditional.csv")
    resid_kurt.to_csv(OUTPUT_DIR / "kurtosis_residual.csv")

    plot_kurtosis(kurt_stats,             "kurtosis_analysis.png")
    plot_rolling_kurtosis(kurt_stats["rolling"], "rolling_kurtosis.png")

    # Summary ---------------------------------------------------------------
    print(f"\n{sep}")
    print("SUMMARY - Significant speed/color signals")
    print(sep)

    for order in ["speed", "color"]:
        sig = static4[static4[f"t_{order}"].abs() > 1.96][
            [order, f"t_{order}"]
        ].sort_values(f"t_{order}", key=abs, ascending=False)
        print(f"\n  {order.upper()} (|t|>1.96, {len(sig)} signals):")
        if sig.empty:
            print("  None.")
        else:
            print(sig.round(5).to_string())

    print(f"\n  All outputs -> {OUTPUT_DIR}")
    print(f"  Charts      -> {FIGURES_DIR}")

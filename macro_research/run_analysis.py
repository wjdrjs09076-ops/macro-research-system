"""
run_analysis.py - Sector Macro Sensitivity Analysis (main runner)

Usage:
    cd macro_research
    python run_analysis.py           # uses cache if available
    python run_analysis.py --force   # re-fetches all data

Output structure:
    output/
    ├── sector_returns.parquet
    ├── macro_changes.parquet
    ├── garch_vol.parquet
    ├── realized_vol.parquet
    ├── vrp_series.csv              VRP time series per sector
    ├── static_results.csv          Full-period delta/gamma regression
    ├── figures/
    │   ├── heatmap_delta.png
    │   ├── heatmap_gamma.png
    │   ├── vrp_time_series.png
    │   └── rolling_<SECTOR>.png
    └── rolling/
        └── <SECTOR>_<MACRO>.csv
"""
from __future__ import annotations

import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from config import OUTPUT_DIR, FIGURES_DIR, ROLLING_DIR, SECTOR_ETFS, MACRO_TICKERS
from data_loader import load_all
from garch_estimator import compute_realized_vol, compute_garch_vol, compute_vrp
from regression_engine import run_static_regression, run_rolling_regression, pivot_summary


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def plot_heatmap(
    val_piv: pd.DataFrame,
    t_piv: pd.DataFrame,
    sig_piv: pd.DataFrame,
    title: str,
    filename: str,
    fmt: str = ".4f",
    cmap_sig: str = "RdYlGn",
):
    """Heatmap: colored cells = statistically significant (|t|>1.96)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    def _star(t: float) -> str:
        a = abs(t)
        if np.isnan(a):  return ""
        if a > 2.576:    return "***"
        if a > 1.960:    return "**"
        if a > 1.645:    return "*"
        return ""

    annot = val_piv.copy().astype(object)
    for r in val_piv.index:
        for c in val_piv.columns:
            v = val_piv.loc[r, c]
            t = t_piv.loc[r, c] if (r in t_piv.index and c in t_piv.columns) else np.nan
            annot.loc[r, c] = "N/A" if np.isnan(v) else f"{v:{fmt}}\n{_star(t)}"

    fig, ax = plt.subplots(figsize=(11, 7))

    sns.heatmap(
        val_piv, ax=ax,
        cmap=cmap_sig, center=0,
        annot=annot, fmt="",
        linewidths=0.5, linecolor="#cccccc",
        cbar_kws={"label": "coefficient (standardized dX)"},
        mask=~sig_piv,
    )
    grey_data = pd.DataFrame(
        np.where(~sig_piv.values, val_piv.values, np.nan),
        index=val_piv.index, columns=val_piv.columns,
    )
    sns.heatmap(
        grey_data, ax=ax,
        cmap="Greys", center=0,
        vmin=val_piv.min().min(), vmax=val_piv.max().max(),
        annot=annot, fmt="",
        linewidths=0.5, linecolor="#cccccc",
        cbar=False, alpha=0.35,
        mask=sig_piv,
    )

    ax.set_title(
        title + "\n[colored = |t|>1.96  |  * p<10%  ** p<5%  *** p<1%]",
        fontsize=12, pad=12,
    )
    ax.set_xlabel("Macro Variable", fontsize=10)
    ax.set_ylabel("Sector ETF", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def plot_rolling(
    rolling_results: dict,
    sector: str,
    macros: list[str],
    filename: str,
):
    """Rolling delta and gamma for one sector across all macro variables."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    n = len(macros)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3.2 * n))
    if n == 1:
        axes = [axes]

    for i, macro in enumerate(macros):
        key = (sector, macro)
        if key not in rolling_results:
            continue
        df = rolling_results[key]

        for j, (coef, color, label) in enumerate(
            [("delta", "steelblue", "delta (linear)"),
             ("gamma", "darkorange", "gamma (convexity)")]
        ):
            ax = axes[i][j]
            df[coef].plot(ax=ax, color=color, linewidth=0.9)
            ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
            ax.fill_between(df.index, df[coef], 0,
                            where=df[coef] > 0, alpha=0.12, color="green")
            ax.fill_between(df.index, df[coef], 0,
                            where=df[coef] <= 0, alpha=0.12, color="red")
            ax.set_title(f"{sector} x {macro} -- {label} (63D rolling)", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    plt.suptitle(
        f"Rolling Delta/Gamma: {sector} ({SECTOR_ETFS.get(sector, '')})",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=130, bbox_inches="tight")
    plt.close()


def plot_vrp(vrp: pd.DataFrame, filename: str):
    """Sector VRP time series (21-day smoothed)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    for col in vrp.columns:
        vrp[col].rolling(21).mean().plot(ax=ax, linewidth=1.0, label=col, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(
        "Sector Variance Risk Premium  (GARCH sigma - Realized Vol, 21-day smoothed)\n"
        "VRP > 0 -> market overpays for vol insurance",
        fontsize=11,
    )
    ax.set_ylabel("VRP (annualized vol, decimal)")
    ax.legend(loc="upper right", fontsize=7, ncol=4)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(force_refetch: bool = False):
    for d in [OUTPUT_DIR, FIGURES_DIR, ROLLING_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    sep = "=" * 62

    # STEP 1: Data -------------------------------------------------------
    print(sep)
    print("STEP 1 - Data loading")
    print(sep)

    cache_ret  = OUTPUT_DIR / "sector_returns.parquet"
    cache_mchg = OUTPUT_DIR / "macro_changes.parquet"
    cache_px   = OUTPUT_DIR / "sector_prices.parquet"
    cache_mlvl = OUTPUT_DIR / "macro_levels.parquet"

    if not force_refetch and cache_ret.exists() and cache_mchg.exists():
        print("  [CACHE HIT] Loading from parquet...")
        returns   = pd.read_parquet(cache_ret)
        macro_chg = pd.read_parquet(cache_mchg)
    else:
        returns, macro_chg, prices, macro_lvl = load_all()
        returns.to_parquet(cache_ret)
        macro_chg.to_parquet(cache_mchg)
        prices.to_parquet(cache_px)
        macro_lvl.to_parquet(cache_mlvl)

    print(f"  Returns  : {returns.index[0].date()} to {returns.index[-1].date()}"
          f"  ({len(returns)} days, {returns.shape[1]} tickers)")
    print(f"  Macro chg: {macro_chg.shape[1]} vars  {list(macro_chg.columns)}")

    # STEP 2: Volatility -------------------------------------------------
    print(f"\n{sep}")
    print("STEP 2 - Volatility: GARCH(1,1) + Realized Vol + VRP")
    print(sep)

    cache_gv = OUTPUT_DIR / "garch_vol.parquet"
    cache_rv = OUTPUT_DIR / "realized_vol.parquet"

    if not force_refetch and cache_gv.exists():
        print("  [CACHE HIT] Loading GARCH vol from parquet...")
        garch_vol    = pd.read_parquet(cache_gv)
        realized_vol = pd.read_parquet(cache_rv)
    else:
        print("  Computing realized vol (20D rolling)...")
        realized_vol = compute_realized_vol(returns)
        print("  Fitting GARCH(1,1) per sector...")
        garch_vol    = compute_garch_vol(returns)
        garch_vol.to_parquet(cache_gv)
        realized_vol.to_parquet(cache_rv)

    vrp = compute_vrp(garch_vol, realized_vol)
    vrp.to_parquet(OUTPUT_DIR / "vrp.parquet")
    vrp.to_csv(OUTPUT_DIR / "vrp_series.csv")

    print("\n  Latest VRP (annualized, decimal):")
    snap = vrp.dropna(how="all").tail(1).T
    snap.columns = ["VRP"]
    print(snap.sort_values("VRP", ascending=False).round(4).to_string())

    plot_vrp(vrp, "vrp_time_series.png")

    # STEP 3: Static regression ------------------------------------------
    print(f"\n{sep}")
    print("STEP 3 - Static quadratic regression (full period, standardized dX)")
    print(sep)

    static_df = run_static_regression(returns, macro_chg, standardize=True)
    static_df.to_csv(OUTPUT_DIR / "static_results.csv")

    delta_piv, t_delta_piv, sig_delta = pivot_summary(static_df, "delta")
    gamma_piv, t_gamma_piv, sig_gamma = pivot_summary(static_df, "gamma")

    print("\n  -- Delta matrix --")
    print(delta_piv.round(4).to_string())
    print("\n  -- Gamma matrix --")
    print(gamma_piv.round(5).to_string())

    plot_heatmap(
        delta_piv, t_delta_piv, sig_delta,
        title="Sector Delta to Macro Variables (full period)",
        filename="heatmap_delta.png",
        fmt=".4f",
        cmap_sig="RdYlGn",
    )
    plot_heatmap(
        gamma_piv, t_gamma_piv, sig_gamma,
        title="Sector Gamma (Convexity) to Macro Variables (full period)",
        filename="heatmap_gamma.png",
        fmt=".5f",
        cmap_sig="PuOr",
    )

    # STEP 4: Rolling regression -----------------------------------------
    print(f"\n{sep}")
    print("STEP 4 - Rolling regression (63D window)")
    print(sep)

    rolling_results = run_rolling_regression(returns, macro_chg, standardize=True)

    for (sector, macro), df in rolling_results.items():
        df.to_csv(ROLLING_DIR / f"{sector}_{macro}.csv")

    sectors = [c for c in returns.columns if c != "SPY"]
    macros  = list(macro_chg.columns)
    for sector in sectors:
        plot_rolling(rolling_results, sector, macros, f"rolling_{sector}.png")

    print(f"\n  Rolling charts saved for {len(sectors)} sectors -> {FIGURES_DIR}")

    # STEP 5: Summary ----------------------------------------------------
    print(f"\n{sep}")
    print("RESEARCH SUMMARY")
    print(sep)

    sig = static_df[static_df["t_gamma"].abs() > 1.96][
        ["gamma", "t_gamma", "delta", "t_delta", "r2"]
    ].sort_values("t_gamma", key=abs, ascending=False)

    if sig.empty:
        print("\n  Significant Gamma (|t|>1.96): none found")
    else:
        print(f"\n  Significant Gamma signals ({len(sig)}, |t_gamma|>1.96):")
        print(sig.round(5).to_string())

    print(f"\n  All outputs  -> {OUTPUT_DIR}")
    print(f"  Charts       -> {FIGURES_DIR}")
    print(f"  Rolling CSV  -> {ROLLING_DIR}")


if __name__ == "__main__":
    force = "--force" in sys.argv
    main(force_refetch=force)

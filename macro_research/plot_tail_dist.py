"""
plot_tail_dist.py - Tail distribution visualization (simple)

3 panels per sector:
  Left  : Return distribution histogram + KDE + normal fit overlay
  Middle: QQ-plot vs normal (tail deviation = fat tail)
  Right : GPD tail fit on left tail exceedances
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import scipy.stats as ss

from config import OUTPUT_DIR, FIGURES_DIR, SECTOR_ETFS


def plot_tail_distributions(returns: pd.DataFrame, gpd_df: pd.DataFrame):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sectors = [c for c in returns.columns if c != "SPY"]

    fig = plt.figure(figsize=(18, len(sectors) * 3.8))
    gs  = gridspec.GridSpec(len(sectors), 3, figure=fig,
                            hspace=0.55, wspace=0.35)

    for row_i, sec in enumerate(sectors):
        r = returns[sec].dropna().values
        xi    = float(gpd_df.loc[sec, "xi"])
        sigma = float(gpd_df.loc[sec, "sigma"])
        u_thr = float(gpd_df.loc[sec, "u"])
        var99 = float(gpd_df.loc[sec, "var99"])
        es99  = float(gpd_df.loc[sec, "es99"])

        mu, std = r.mean(), r.std()

        # ── Panel 1: Histogram + KDE + Normal overlay ──────────────────────
        ax1 = fig.add_subplot(gs[row_i, 0])

        # histogram (log scale y for tail visibility)
        counts, bins, _ = ax1.hist(r, bins=120, density=True,
                                   color="#90CAF9", alpha=0.55,
                                   edgecolor="none", label="Empirical")

        # KDE
        kde = ss.gaussian_kde(r, bw_method=0.3)
        x_range = np.linspace(r.min(), r.max(), 500)
        ax1.plot(x_range, kde(x_range), color="#1565C0",
                 linewidth=1.4, label="KDE")

        # Normal fit
        ax1.plot(x_range, ss.norm.pdf(x_range, mu, std),
                 color="#C62828", linewidth=1.2, linestyle="--", label="Normal fit")

        # VaR / ES lines
        ax1.axvline(-var99, color="darkorange", linewidth=1.0, linestyle=":",
                    label=f"EVT VaR99={var99:.3f}")
        ax1.axvline(-es99,  color="red",        linewidth=1.0, linestyle=":",
                    label=f"EVT ES99={es99:.3f}")

        ax1.set_yscale("log")
        ax1.set_xlim(np.percentile(r, 0.1), np.percentile(r, 99.9))
        ax1.set_title(f"{sec} — Return Distribution\n(log y-scale)",
                      fontsize=8, pad=4)
        ax1.set_xlabel("Daily log return", fontsize=7)
        ax1.set_ylabel("Density (log)", fontsize=7)
        ax1.tick_params(labelsize=6)
        ax1.legend(fontsize=5.5, loc="upper left")

        # ── Panel 2: QQ-plot vs Normal ──────────────────────────────────────
        ax2 = fig.add_subplot(gs[row_i, 1])

        (osm, osr), (slope, intercept, _) = ss.probplot(r, dist="norm")
        ax2.scatter(osm, osr, s=3, alpha=0.4, color="#1565C0")
        # reference line
        line_x = np.array([osm[0], osm[-1]])
        ax2.plot(line_x, slope * line_x + intercept,
                 color="#C62828", linewidth=1.0, linestyle="--", label="Normal ref")

        # shade tail deviation zones
        tail_mask_lo = osm < -2
        tail_mask_hi = osm > 2
        ax2.scatter(osm[tail_mask_lo], osr[tail_mask_lo],
                    s=8, color="red",   alpha=0.7, zorder=3, label="Fat lower tail")
        ax2.scatter(osm[tail_mask_hi], osr[tail_mask_hi],
                    s=8, color="green", alpha=0.7, zorder=3, label="Fat upper tail")

        skew = ss.skew(r)
        kurt = ss.kurtosis(r)
        ax2.set_title(f"{sec} — Normal QQ-plot\nskew={skew:.2f}  kurt={kurt:.2f}",
                      fontsize=8, pad=4)
        ax2.set_xlabel("Normal quantile", fontsize=7)
        ax2.set_ylabel("Empirical quantile", fontsize=7)
        ax2.tick_params(labelsize=6)
        ax2.legend(fontsize=5.5)

        # ── Panel 3: GPD tail fit ───────────────────────────────────────────
        ax3 = fig.add_subplot(gs[row_i, 2])

        losses    = -r
        exceedances = np.sort(losses[losses > u_thr] - u_thr)
        n_e = len(exceedances)

        if n_e > 0 and not np.isnan(xi):
            probs = (np.arange(1, n_e + 1)) / (n_e + 1)
            if abs(xi) < 1e-8:
                gpd_q = -sigma * np.log(1 - probs)
            else:
                gpd_q = (sigma / xi) * ((1 - probs) ** (-xi) - 1)

            ax3.scatter(gpd_q, exceedances, s=6, alpha=0.55, color="#1565C0",
                        label="Observed exceedances")
            max_v = max(gpd_q.max(), exceedances.max()) * 1.05
            ax3.plot([0, max_v], [0, max_v], "r--", linewidth=1.0, label="Perfect fit")

            # Shade over-fit region (empirical > GPD -> fatter than GPD)
            over_mask = exceedances > gpd_q
            ax3.scatter(gpd_q[over_mask], exceedances[over_mask],
                        s=10, color="red", alpha=0.7, zorder=3, label="Fatter than GPD")

        ax3.set_title(
            f"{sec} — GPD Tail Fit (left tail)\n"
            f"xi={xi:.3f}  threshold={u_thr:.4f}",
            fontsize=8, pad=4,
        )
        ax3.set_xlabel("GPD theoretical quantile", fontsize=7)
        ax3.set_ylabel("Empirical loss quantile", fontsize=7)
        ax3.tick_params(labelsize=6)
        ax3.legend(fontsize=5.5)

    # Super-title
    fig.suptitle(
        "Sector Tail Distribution Analysis\n"
        "Left: log-scale histogram  |  Middle: QQ-plot vs Normal  |  Right: GPD tail fit",
        fontsize=12, y=1.002,
    )

    out = FIGURES_DIR / "tail_distributions_full.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Summary panel — all sectors on one overview chart
# ---------------------------------------------------------------------------

def plot_tail_overview(returns: pd.DataFrame, gpd_df: pd.DataFrame):
    """Single-page overview: KDE comparison + QQ deviation + CF VaR bar."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sectors = [c for c in returns.columns if c != "SPY"]

    cf_df = pd.read_csv(OUTPUT_DIR / "tail_cf_var.csv", index_col=0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ── Left: KDE overlay (all sectors, left tail zoom) ────────────────────
    ax = axes[0]
    colors = plt.cm.tab20(np.linspace(0, 1, len(sectors)))
    for sec, c in zip(sectors, colors):
        r = returns[sec].dropna().values
        kde = ss.gaussian_kde(r, bw_method=0.25)
        x = np.linspace(-0.12, 0.0, 300)
        ax.plot(x, kde(x), color=c, linewidth=1.3, label=sec, alpha=0.85)
    # Normal reference
    r_all = returns[sectors].stack().values
    mu_a, std_a = r_all.mean(), r_all.std()
    ax.plot(np.linspace(-0.12, 0.0, 300),
            ss.norm.pdf(np.linspace(-0.12, 0.0, 300), mu_a, std_a),
            "k--", linewidth=1.5, label="Normal ref")
    ax.set_title("Left Tail KDE (all sectors)\nHigher density = fatter tail",
                 fontsize=10)
    ax.set_xlabel("Daily log return", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=6.5, ncol=2)
    ax.set_xlim(-0.12, 0.0)

    # ── Middle: GPD tail index xi comparison ───────────────────────────────
    ax = axes[1]
    xi_vals = gpd_df["xi"].reindex(sectors)
    bar_colors = ["#C62828" if v > 0.2 else "#1565C0" for v in xi_vals]
    bars = ax.bar(sectors, xi_vals, color=bar_colors, edgecolor="black",
                  linewidth=0.5, alpha=0.85)
    ax.axhline(0,   color="black", linewidth=0.7)
    ax.axhline(0.5, color="red",   linewidth=0.7, linestyle="--",
               label="xi=0.5 (infinite variance)")
    ax.axhline(0.25, color="orange", linewidth=0.7, linestyle=":",
               label="xi=0.25 (reference)")
    ax.set_title("GPD Tail Index xi\nRed = xi>0.2  |  All > 0 = fat tail",
                 fontsize=10)
    ax.set_ylabel("Tail index xi", fontsize=9)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.legend(fontsize=7)

    # ── Right: CF VaR vs Normal VaR ratio ──────────────────────────────────
    ax = axes[2]
    ratio = cf_df["cf_over_normal"].reindex(sectors)
    bar_colors2 = ["#C62828" if v > 2.5 else "#FB8C00" if v > 2.0 else "#1565C0"
                   for v in ratio]
    ax.bar(sectors, ratio, color=bar_colors2, edgecolor="black",
           linewidth=0.5, alpha=0.85)
    ax.axhline(1.0, color="black",  linewidth=0.7, linestyle="--", label="Normal (ratio=1)")
    ax.axhline(2.0, color="orange", linewidth=0.7, linestyle=":",  label="2x threshold")
    ax.axhline(3.0, color="red",    linewidth=0.7, linestyle=":",  label="3x threshold")
    ax.set_title("Cornish-Fisher / Normal VaR ratio\nHow much normal dist underestimates risk",
                 fontsize=10)
    ax.set_ylabel("CF VaR(99%) / Normal VaR(99%)", fontsize=9)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.legend(fontsize=7)

    plt.suptitle("Tail Risk Overview — All Sectors", fontsize=13, y=1.02)
    plt.tight_layout()
    out = FIGURES_DIR / "tail_overview.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    returns = pd.read_parquet(OUTPUT_DIR / "sector_returns.parquet")
    gpd_df  = pd.read_csv(OUTPUT_DIR / "tail_gpd.csv", index_col=0)

    print("Plotting individual sector tail distributions...")
    plot_tail_distributions(returns, gpd_df)

    print("Plotting overview panel...")
    plot_tail_overview(returns, gpd_df)

    print("Done.")

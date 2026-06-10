"""
tail_analysis.py - Deep tail risk analysis (4 methods)

1. GPD / EVT  : Generalized Pareto Distribution tail fitting
                Tail index xi, EVT-based VaR(99%), ES(99%)
2. Quantile   : Quantile regression at tau = 0.01, 0.05, 0.25, 0.75, 0.95, 0.99
                Asymmetry: lower tail sensitivity vs upper tail sensitivity
3. Tail Dep.  : Empirical tail dependence lambda_L / lambda_U
                Sector x Sector and Sector x Macro matrices
4. CF VaR     : Cornish-Fisher modified VaR accounting for skewness and kurtosis
                Quantifies how much normal-dist VaR underestimates true tail risk
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
from statsmodels.regression.quantile_regression import QuantReg

from config import OUTPUT_DIR, FIGURES_DIR, SECTOR_ETFS, MACRO_TICKERS


# ---------------------------------------------------------------------------
# 1. GPD / Extreme Value Theory
# ---------------------------------------------------------------------------

def fit_gpd(losses: np.ndarray, threshold_pct: float = 0.10) -> dict:
    """
    Fit Generalized Pareto Distribution to left-tail exceedances.

    losses    : positive values representing losses (-returns for left tail)
    threshold : u = empirical quantile at threshold_pct (e.g. 10% worst days)

    Returns: xi (shape), sigma (scale), u (threshold),
             var99, es99 (EVT-based), n_exceed
    """
    losses = losses[~np.isnan(losses)]
    u = np.quantile(losses, 1 - threshold_pct)
    exceedances = losses[losses > u] - u

    n = len(losses)
    n_e = len(exceedances)

    if n_e < 20:
        return {k: np.nan for k in ["xi","sigma","u","var99","es99","n_exceed","n_total"]}

    try:
        xi, loc, sigma = ss.genpareto.fit(exceedances, floc=0)
    except Exception:
        return {k: np.nan for k in ["xi","sigma","u","var99","es99","n_exceed","n_total"]}

    # EVT VaR and ES at 99% (1% exceedance probability)
    p      = 0.01
    zeta_u = n_e / n  # fraction of observations above threshold

    if abs(xi) < 1e-8:  # xi ~ 0: exponential tail
        var99 = u + sigma * np.log(zeta_u / p)
        es99  = var99 + sigma
    else:
        var99 = u + (sigma / xi) * ((zeta_u / p) ** xi - 1)
        es99  = (var99 + sigma - xi * u) / (1 - xi) if xi < 1 else np.nan

    return {
        "xi": xi, "sigma": sigma, "u": u,
        "var99": var99, "es99": es99,
        "n_exceed": n_e, "n_total": n,
    }


def run_gpd_all(returns: pd.DataFrame) -> pd.DataFrame:
    """Fit GPD to left tail of each sector's daily returns."""
    sectors = [c for c in returns.columns if c != "SPY"]
    rows = []
    for sec in sectors:
        losses = -returns[sec].dropna().values  # flip sign: loss = positive
        res = fit_gpd(losses)
        rows.append({"sector": sec, **res})
    df = pd.DataFrame(rows).set_index("sector")

    # Also compute historical (empirical) VaR/ES for comparison
    for sec in sectors:
        r = returns[sec].dropna()
        df.loc[sec, "var99_hist"] = float(-np.percentile(r, 1))
        es_mask = r <= np.percentile(r, 1)
        df.loc[sec, "es99_hist"]  = float(-r[es_mask].mean()) if es_mask.sum() > 0 else np.nan

    return df


def plot_gpd_fits(returns: pd.DataFrame, gpd_df: pd.DataFrame, filename: str):
    """QQ-plot of GPD fit vs empirical tail for each sector."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sectors = [c for c in returns.columns if c != "SPY"]
    n = len(sectors)
    cols = 3
    rows_n = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows_n, cols, figsize=(15, 4 * rows_n))
    axes_flat = axes.flatten()

    for i, sec in enumerate(sectors):
        ax = axes_flat[i]
        losses = -returns[sec].dropna().values
        row = gpd_df.loc[sec]
        u, xi, sigma = row["u"], row["xi"], row["sigma"]

        if np.isnan(xi):
            ax.set_visible(False)
            continue

        exceed = np.sort(losses[losses > u] - u)
        n_e = len(exceed)
        probs = (np.arange(1, n_e + 1)) / (n_e + 1)

        if abs(xi) < 1e-8:
            gpd_q = -sigma * np.log(1 - probs)
        else:
            gpd_q = (sigma / xi) * ((1 - probs) ** (-xi) - 1)

        ax.scatter(gpd_q, exceed, s=8, alpha=0.6, color="steelblue")
        max_v = max(gpd_q.max(), exceed.max())
        ax.plot([0, max_v], [0, max_v], "r--", linewidth=0.8)
        ax.set_title(
            f"{sec}\nxi={xi:.3f}  sigma={sigma:.4f}\n"
            f"VaR99={row['var99']:.3f}  ES99={row['es99']:.3f}",
            fontsize=8,
        )
        ax.set_xlabel("GPD quantile", fontsize=7)
        ax.set_ylabel("Empirical quantile", fontsize=7)
        ax.tick_params(labelsize=6)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle("GPD Tail Fit: QQ-Plot (left tail exceedances)\nxi > 0 = fat tail  |  red line = perfect fit",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


# ---------------------------------------------------------------------------
# 2. Quantile Regression
# ---------------------------------------------------------------------------

QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def run_quantile_regression(
    returns: pd.DataFrame,
    macro_changes: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each (sector, macro, quantile): fit R_tau = a + b*dX.
    Returns long-form DataFrame with columns: sector, macro, tau, alpha, beta, t_beta, pseudo_r2
    """
    sectors = [c for c in returns.columns if c != "SPY"]
    macros  = [c for c in macro_changes.columns if c in MACRO_TICKERS]

    common = returns.index.intersection(macro_changes.index)
    R = returns.loc[common]
    M = macro_changes.loc[common].apply(lambda s: (s - s.mean()) / (s.std() or 1))

    rows = []
    total = len(sectors) * len(macros)
    cnt = 0

    for sec in sectors:
        for mac in macros:
            cnt += 1
            print(f"  QR [{cnt:>2}/{total}] {sec} x {mac}...", flush=True)
            y = R[sec].values
            x = M[mac].values
            mask = ~(np.isnan(y) | np.isnan(x))
            y_c, x_c = y[mask], x[mask]

            if len(y_c) < 50:
                continue

            X_const = sm.add_constant(x_c.reshape(-1, 1), has_constant="add")

            for tau in QUANTILES:
                try:
                    res = QuantReg(y_c, X_const).fit(q=tau, max_iter=2000)
                    rows.append({
                        "sector": sec, "macro": mac, "tau": tau,
                        "alpha":     res.params[0],
                        "beta":      res.params[1],
                        "t_beta":    res.tvalues[1],
                        "pseudo_r2": res.prsquared,
                    })
                except Exception:
                    pass

    return pd.DataFrame(rows)


def plot_quantile_coefs(qr_df: pd.DataFrame, macro: str, filename: str):
    """
    Plot beta(tau) across quantiles for each sector — one macro variable.
    Asymmetry: if beta(0.05) != beta(0.95), macro affects tails differently.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    subset  = qr_df[qr_df["macro"] == macro]
    sectors = sorted(subset["sector"].unique())

    n = len(sectors)
    cols = 3
    rows_n = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows_n, cols, figsize=(15, 3.5 * rows_n))
    axes_flat = axes.flatten()

    for i, sec in enumerate(sectors):
        ax = axes_flat[i]
        df_s = subset[subset["sector"] == sec].sort_values("tau")

        ax.plot(df_s["tau"], df_s["beta"], marker="o", markersize=4,
                color="steelblue", linewidth=1.2, label="beta(tau)")
        ax.fill_between(df_s["tau"], df_s["beta"], 0,
                        where=df_s["beta"] > 0, alpha=0.10, color="green")
        ax.fill_between(df_s["tau"], df_s["beta"], 0,
                        where=df_s["beta"] <= 0, alpha=0.10, color="red")
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.axvline(0.5, color="grey", linewidth=0.5, linestyle=":")

        # Highlight lower vs upper tail asymmetry
        b_low  = df_s[df_s["tau"] == 0.05]["beta"].values
        b_high = df_s[df_s["tau"] == 0.95]["beta"].values
        if len(b_low) and len(b_high):
            asym = float(b_low[0]) - float(b_high[0])
            ax.set_title(f"{sec} ({SECTOR_ETFS.get(sec,'')})\nasymmetry(5%-95%)={asym:.5f}",
                         fontsize=8)
        else:
            ax.set_title(f"{sec}", fontsize=8)

        ax.set_xlabel("quantile tau", fontsize=7)
        ax.set_ylabel("beta(tau)", fontsize=7)
        ax.tick_params(labelsize=6)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle(
        f"Quantile Regression: {macro} -> sector returns\n"
        "If beta(0.05) < beta(0.95): macro hits harder in downturns (asymmetric)",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def compute_tail_asymmetry(qr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tail asymmetry = beta(tau=0.05) - beta(tau=0.95) for each (sector, macro).
    Negative: lower tail beta < upper tail beta -> macro hits harder in downturns.
    Positive: macro hits harder in upturns.
    """
    rows = []
    for (sec, mac), grp in qr_df.groupby(["sector", "macro"]):
        b_low  = grp[grp["tau"] == 0.05]["beta"].values
        b_high = grp[grp["tau"] == 0.95]["beta"].values
        t_low  = grp[grp["tau"] == 0.05]["t_beta"].values
        t_high = grp[grp["tau"] == 0.95]["t_beta"].values
        b_med  = grp[grp["tau"] == 0.50]["beta"].values

        if len(b_low) and len(b_high) and len(b_med):
            rows.append({
                "sector": sec, "macro": mac,
                "beta_005":   float(b_low[0]),
                "beta_050":   float(b_med[0]),
                "beta_095":   float(b_high[0]),
                "t_005":      float(t_low[0])  if len(t_low)  else np.nan,
                "t_095":      float(t_high[0]) if len(t_high) else np.nan,
                "asymmetry":  float(b_low[0]) - float(b_high[0]),
            })
    return pd.DataFrame(rows).set_index(["sector", "macro"])


# ---------------------------------------------------------------------------
# 3. Tail Dependence
# ---------------------------------------------------------------------------

def empirical_tail_dep(x: np.ndarray, y: np.ndarray, u: float = 0.05) -> tuple[float, float]:
    """
    Empirical lower and upper tail dependence at probability level u.

    lambda_L = P(X < Q_X(u) | Y < Q_Y(u))  -- co-crash probability
    lambda_U = P(X > Q_X(1-u) | Y > Q_Y(1-u))  -- co-rally probability
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)

    qx_lo, qy_lo = np.quantile(x, u),     np.quantile(y, u)
    qx_hi, qy_hi = np.quantile(x, 1 - u), np.quantile(y, 1 - u)

    lambda_l = np.mean((x < qx_lo) & (y < qy_lo)) / u
    lambda_u = np.mean((x > qx_hi) & (y > qy_hi)) / u

    return float(lambda_l), float(lambda_u)


def compute_tail_dep_matrix(
    returns: pd.DataFrame,
    macro_changes: pd.DataFrame,
    u: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns four matrices:
        sector_lower : sector x sector lower tail dependence
        sector_upper : sector x sector upper tail dependence
        macro_lower  : sector x macro lower tail dependence
        macro_upper  : sector x macro upper tail dependence
    """
    sectors = [c for c in returns.columns if c != "SPY"]
    macros  = [c for c in macro_changes.columns if c in MACRO_TICKERS]

    common = returns.index.intersection(macro_changes.index)
    R = returns.loc[common]
    M = macro_changes.loc[common]

    # Sector x Sector
    sec_lo = pd.DataFrame(index=sectors, columns=sectors, dtype=float)
    sec_hi = pd.DataFrame(index=sectors, columns=sectors, dtype=float)
    for i, s1 in enumerate(sectors):
        for s2 in sectors[i:]:
            lo, hi = empirical_tail_dep(R[s1].values, R[s2].values, u)
            sec_lo.loc[s1, s2] = sec_lo.loc[s2, s1] = lo
            sec_hi.loc[s1, s2] = sec_hi.loc[s2, s1] = hi

    # Sector x Macro
    mac_lo = pd.DataFrame(index=sectors, columns=macros, dtype=float)
    mac_hi = pd.DataFrame(index=sectors, columns=macros, dtype=float)
    for sec in sectors:
        for mac in macros:
            lo, hi = empirical_tail_dep(R[sec].values, M[mac].values, u)
            mac_lo.loc[sec, mac] = lo
            mac_hi.loc[sec, mac] = hi

    return sec_lo, sec_hi, mac_lo, mac_hi


def plot_tail_dep_heatmap(
    matrix: pd.DataFrame,
    title: str,
    filename: str,
    cmap: str = "YlOrRd",
    vmin: float = 0.0,
    vmax: float = 1.0,
):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    m = matrix.astype(float)
    sns.heatmap(m, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                annot=True, fmt=".2f", linewidths=0.4,
                cbar_kws={"label": "tail dependence lambda"})
    ax.set_title(title + "\n(u=0.05, random independence = 0.05)", fontsize=10, pad=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


# ---------------------------------------------------------------------------
# 4. Cornish-Fisher Modified VaR
# ---------------------------------------------------------------------------

def cornish_fisher_var(
    returns: pd.Series,
    alpha: float = 0.01,
) -> dict:
    """
    Modified VaR via Cornish-Fisher expansion:
        z_cf = z + (z^2-1)*S/6 + (z^3-3z)*K/24 - (2z^3-5z)*S^2/36
        VaR_CF = mu + sigma * z_cf

    where S = skewness, K = excess kurtosis, z = normal quantile at alpha.

    Returns: var_normal, var_cf, ratio (how much CF > normal), S, K
    """
    r = returns.dropna().values
    mu    = np.mean(r)
    sigma = np.std(r, ddof=1)
    S     = float(ss.skew(r))
    K     = float(ss.kurtosis(r))  # excess kurtosis (Fisher)
    z     = ss.norm.ppf(alpha)     # negative for alpha < 0.5

    z_cf = (z
            + (z**2 - 1) * S / 6
            + (z**3 - 3*z) * K / 24
            - (2*z**3 - 5*z) * S**2 / 36)

    var_normal = -(mu + sigma * z)
    var_cf     = -(mu + sigma * z_cf)
    var_hist   = float(-np.percentile(r, alpha * 100))

    return {
        "var_normal": var_normal,
        "var_cf":     var_cf,
        "var_hist":   var_hist,
        "cf_over_normal": var_cf / var_normal if var_normal != 0 else np.nan,
        "skewness":   S,
        "exc_kurtosis": K,
    }


def run_cf_var_all(returns: pd.DataFrame, alpha: float = 0.01) -> pd.DataFrame:
    sectors = [c for c in returns.columns if c != "SPY"]
    rows = []
    for sec in sectors:
        res = cornish_fisher_var(returns[sec], alpha)
        rows.append({"sector": sec, **res})
    return pd.DataFrame(rows).set_index("sector")


def plot_var_comparison(cf_df: pd.DataFrame, gpd_df: pd.DataFrame, filename: str):
    """Bar chart: Normal VaR vs CF VaR vs EVT VaR vs Historical VaR per sector."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sectors = cf_df.index.tolist()
    x = np.arange(len(sectors))
    w = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.bar(x - 1.5*w, cf_df["var_normal"], w, label="Normal VaR(99%)",
           color="#90CAF9", edgecolor="black", linewidth=0.4)
    ax.bar(x - 0.5*w, cf_df["var_cf"],     w, label="Cornish-Fisher VaR(99%)",
           color="#1565C0", edgecolor="black", linewidth=0.4)
    ax.bar(x + 0.5*w, gpd_df["var99"],     w, label="EVT/GPD VaR(99%)",
           color="#C62828", edgecolor="black", linewidth=0.4)
    ax.bar(x + 1.5*w, cf_df["var_hist"],   w, label="Historical VaR(99%)",
           color="#558B2F", edgecolor="black", linewidth=0.4, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(sectors, rotation=30, fontsize=9)
    ax.set_ylabel("VaR (daily loss, decimal)")
    ax.set_title(
        "VaR(99%) Comparison: Normal vs Cornish-Fisher vs EVT/GPD vs Historical\n"
        "Higher bar = model says tail is fatter than normal dist",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5)
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

    sep = "=" * 62

    returns   = pd.read_parquet(OUTPUT_DIR / "sector_returns.parquet")
    macro_chg = pd.read_parquet(OUTPUT_DIR / "macro_changes.parquet")

    # ── 1. GPD / EVT ─────────────────────────────────────────────────────────
    print(sep)
    print("STEP 1 - GPD / Extreme Value Theory tail fitting")
    print(sep)

    gpd_df = run_gpd_all(returns)
    gpd_df.to_csv(OUTPUT_DIR / "tail_gpd.csv")

    print("\n  GPD results (xi>0 = fat tail, xi>0.5 = infinite variance):")
    print(gpd_df[["xi","sigma","u","var99","es99","var99_hist","es99_hist"]].round(4).to_string())
    plot_gpd_fits(returns, gpd_df, "tail_gpd_qqplot.png")

    # ── 2. Quantile Regression ───────────────────────────────────────────────
    print(f"\n{sep}")
    print("STEP 2 - Quantile regression (tau = 0.01 to 0.99)")
    print(sep)

    qr_df = run_quantile_regression(returns, macro_chg)
    qr_df.to_csv(OUTPUT_DIR / "tail_quantile_regression.csv", index=False)

    asym_df = compute_tail_asymmetry(qr_df)
    asym_df.to_csv(OUTPUT_DIR / "tail_asymmetry.csv")

    print("\n  Tail asymmetry = beta(tau=0.05) - beta(tau=0.95)")
    print("  Negative = macro hits harder in downturns:")
    asym_piv = asym_df["asymmetry"].unstack("macro")
    print(asym_piv.round(5).to_string())

    for mac in macro_chg.columns:
        plot_quantile_coefs(qr_df, mac, f"qr_coef_{mac}.png")

    # ── 3. Tail Dependence ───────────────────────────────────────────────────
    print(f"\n{sep}")
    print("STEP 3 - Tail dependence (u=0.05)")
    print(sep)

    sec_lo, sec_hi, mac_lo, mac_hi = compute_tail_dep_matrix(returns, macro_chg)

    sec_lo.to_csv(OUTPUT_DIR / "tail_dep_sector_lower.csv")
    sec_hi.to_csv(OUTPUT_DIR / "tail_dep_sector_upper.csv")
    mac_lo.to_csv(OUTPUT_DIR / "tail_dep_macro_lower.csv")
    mac_hi.to_csv(OUTPUT_DIR / "tail_dep_macro_upper.csv")

    print("\n  Sector x Sector lower tail dependence (co-crash):")
    print(sec_lo.astype(float).round(3).to_string())

    print("\n  Sector x Macro lower tail dependence:")
    print(mac_lo.astype(float).round(3).to_string())

    plot_tail_dep_heatmap(sec_lo.astype(float), "Sector x Sector Lower Tail Dependence (co-crash, u=0.05)",
                          "tail_dep_sector_lower.png", vmax=0.5)
    plot_tail_dep_heatmap(sec_hi.astype(float), "Sector x Sector Upper Tail Dependence (co-rally, u=0.05)",
                          "tail_dep_sector_upper.png", vmax=0.5)
    plot_tail_dep_heatmap(mac_lo.astype(float), "Sector x Macro Lower Tail Dependence (u=0.05)",
                          "tail_dep_macro_lower.png", vmax=0.5)

    # ── 4. Cornish-Fisher VaR ────────────────────────────────────────────────
    print(f"\n{sep}")
    print("STEP 4 - Cornish-Fisher modified VaR(99%)")
    print(sep)

    cf_df = run_cf_var_all(returns, alpha=0.01)
    cf_df.to_csv(OUTPUT_DIR / "tail_cf_var.csv")

    print("\n  VaR(99%) comparison:")
    disp = cf_df[["var_normal","var_cf","var_hist","cf_over_normal","skewness","exc_kurtosis"]]
    print(disp.sort_values("cf_over_normal", ascending=False).round(4).to_string())

    plot_var_comparison(cf_df, gpd_df, "tail_var_comparison.png")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("TAIL RISK SUMMARY")
    print(sep)

    print("\n  GPD tail index xi (>0 = fat tail, >0.5 = infinite variance):")
    xi_sorted = gpd_df["xi"].sort_values(ascending=False)
    for sec, xi in xi_sorted.items():
        flag = " *** INFINITE VARIANCE" if xi > 0.5 else (" ** fat tail" if xi > 0 else "")
        print(f"  {sec:5s}: xi={xi:+.4f}{flag}")

    print("\n  Biggest CF/Normal VaR ratio (fat tail underestimation by normal dist):")
    ratio = cf_df["cf_over_normal"].sort_values(ascending=False)
    for sec, r in ratio.items():
        print(f"  {sec:5s}: CF VaR = {r:.3f}x normal VaR")

    print("\n  Most tail-dependent sector pairs (lower tail, co-crash):")
    lo = sec_lo.astype(float)
    lo_stack = lo.where(np.triu(np.ones(lo.shape), k=1).astype(bool)).stack()
    top5 = lo_stack.sort_values(ascending=False).head(5)
    for (s1, s2), v in top5.items():
        print(f"  {s1} x {s2}: lambda_L = {v:.3f}")

    print(f"\n  All outputs -> {OUTPUT_DIR}")
    print(f"  Charts      -> {FIGURES_DIR}")

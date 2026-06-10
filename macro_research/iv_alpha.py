"""
iv_alpha.py - IV Spread as alpha signal

Two analyses:

1. Extended contemporaneous regression
   Add delta(IV_spread) to the existing macro regression:
   R_s = a + sum_X[d*dX + g*dX^2] + d_iv*d(IV_spread_s) + g_iv*d(IV_spread_s)^2 + e

   IV_spread_s(t) = CAPM_IV_s(t) - VIX(t)/100
   -> measures sector-specific fear premium above market-wide fear

2. Predictive regression (core alpha test)
   R_s(t+k) = a + b*IV_spread_s(t) + e   for k = 1, 5, 21 days
   R_s(t+k) = a + b*VRP_s(t) + e         for k = 1, 5, 21 days

   If b > 0 and significant: high IV premium predicts positive future return
   -> options market overstates risk -> mean reversion -> alpha

3. Long-short portfolio simulation
   Each day: rank sectors by signal (IV_spread or VRP)
   Long top-3, Short bottom-3 -> equal-weight daily rebalance
   Track cumulative return vs SPY
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import OUTPUT_DIR, FIGURES_DIR, SECTOR_ETFS, OOS_START


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------

def build_iv_signals(
    capm_iv: pd.DataFrame,
    realized_vol: pd.DataFrame,
    macro_levels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        iv_spread : CAPM_IV_sector - VIX/100  (sector IV premium over market)
        vrp       : CAPM_IV_sector - RV_sector (variance risk premium)
    """
    sectors = [c for c in capm_iv.columns if c != "SPY"]

    vix_decimal = macro_levels["VIX"].reindex(capm_iv.index).ffill() / 100.0

    iv_spread = pd.DataFrame(index=capm_iv.index, columns=sectors, dtype=float)
    vrp       = pd.DataFrame(index=capm_iv.index, columns=sectors, dtype=float)

    for sec in sectors:
        iv_spread[sec] = capm_iv[sec] - vix_decimal
        vrp[sec]       = capm_iv[sec] - realized_vol[sec].reindex(capm_iv.index)

    return iv_spread, vrp


# ---------------------------------------------------------------------------
# Part 1: Extended contemporaneous regression
# ---------------------------------------------------------------------------

def run_extended_regression(
    returns: pd.DataFrame,
    macro_changes: pd.DataFrame,
    iv_spread: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each sector, add delta(IV_spread) to the macro regression.

    Model:
        R_s = a + sum_X[d_X*dX + g_X*dX^2] + d_iv*d(IV_spread) + g_iv*d(IV_spread)^2 + e

    Returns DataFrame with {sector: {variable: {alpha,delta,gamma,t_*,r2}}}
    stored as MultiIndex rows.
    """
    sectors = [c for c in returns.columns if c != "SPY"]
    macros  = [c for c in macro_changes.columns]

    # Standardize macro changes
    macro_std = macro_changes.apply(lambda s: (s - s.mean()) / s.std())

    # IV spread first difference (standardized)
    div_spread = iv_spread.diff()
    div_std    = div_spread.apply(lambda s: (s - s.mean()) / (s.std() if s.std() > 0 else 1))

    # Align index
    common = returns.index.intersection(macro_std.index).intersection(div_std.index)
    R  = returns.loc[common]
    M  = macro_std.loc[common]
    IV = div_std.loc[common]

    rows = []
    for sec in sectors:
        y = R[sec].values

        # Build feature matrix: [dX1, dX1^2, dX2, dX2^2, ..., dIV, dIV^2]
        feature_cols = []
        feature_names = []
        for mac in macros:
            x = M[mac].values
            feature_cols += [x, x**2]
            feature_names += [mac, mac + "^2"]
        # IV spread term
        iv_col = IV[sec].values
        feature_cols += [iv_col, iv_col**2]
        feature_names += ["IV_spread", "IV_spread^2"]

        X = np.column_stack(feature_cols)
        mask = ~np.any(np.isnan(np.column_stack([y.reshape(-1,1), X])), axis=1)
        y_c, X_c = y[mask], X[mask]

        if len(y_c) < 50:
            continue

        X_const = sm.add_constant(X_c, has_constant="add")
        try:
            res = sm.OLS(y_c, X_const).fit()
        except Exception:
            continue

        row = {"sector": sec, "n": int(res.nobs), "r2": res.rsquared, "r2_adj": res.rsquared_adj}
        param_names = ["const"] + feature_names
        for i, name in enumerate(param_names):
            row[f"coef_{name}"]  = res.params[i]
            row[f"tval_{name}"]  = res.tvalues[i]
        rows.append(row)

    return pd.DataFrame(rows).set_index("sector")


# ---------------------------------------------------------------------------
# Part 2: Predictive regression
# ---------------------------------------------------------------------------

def run_predictive_regression(
    returns: pd.DataFrame,
    signal: pd.DataFrame,
    horizons: list[int] = [1, 5, 21],
    signal_name: str = "IV_spread",
) -> pd.DataFrame:
    """
    R_s(t -> t+k) = a + b*signal_s(t) + e

    Tests whether signal today predicts forward returns.
    Returns summary: sector x horizon -> {b, t_b, r2, n}
    """
    sectors = [c for c in returns.columns if c in signal.columns]
    rows = []

    for sec in sectors:
        for k in horizons:
            # forward k-day cumulative log return
            fwd_ret = returns[sec].rolling(k).sum().shift(-k)

            sig_s = signal[sec]
            common = sig_s.index.intersection(fwd_ret.index)
            y = fwd_ret.loc[common].values
            x = sig_s.loc[common].values

            mask = ~(np.isnan(y) | np.isnan(x))
            y_c, x_c = y[mask], x[mask]

            if len(y_c) < 50:
                continue

            X_const = sm.add_constant(x_c.reshape(-1, 1), has_constant="add")
            try:
                res = sm.OLS(y_c, X_const).fit()
            except Exception:
                continue

            rows.append({
                "sector":   sec,
                "horizon":  k,
                "signal":   signal_name,
                "b":        res.params[1],
                "t_b":      res.tvalues[1],
                "r2":       res.rsquared,
                "n":        int(res.nobs),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part 3: Long-short portfolio simulation
# ---------------------------------------------------------------------------

def run_longshort_backtest(
    returns: pd.DataFrame,
    signal: pd.DataFrame,
    n_long: int = 3,
    n_short: int = 3,
    signal_name: str = "signal",
) -> pd.Series:
    """
    Each day: long top-n_long sectors, short bottom-n_short sectors by signal.
    Equal-weight within long and short legs.
    Returns daily portfolio return series.
    """
    sectors = [c for c in returns.columns if c in signal.columns]
    common  = returns.index.intersection(signal.index)

    port_rets = []
    dates     = []

    for i in range(1, len(common)):
        date_today = common[i]
        date_prev  = common[i - 1]

        sig_today  = signal.loc[date_prev, sectors].dropna()
        if len(sig_today) < n_long + n_short:
            continue

        ranked  = sig_today.sort_values(ascending=False)
        longs   = ranked.iloc[:n_long].index.tolist()
        shorts  = ranked.iloc[-n_short:].index.tolist()

        ret_today = returns.loc[date_today, sectors]

        long_ret  = ret_today[longs].mean()
        short_ret = ret_today[shorts].mean()
        port_ret  = long_ret - short_ret

        port_rets.append(port_ret)
        dates.append(date_today)

    return pd.Series(port_rets, index=dates, name=signal_name)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_predictive_heatmap(pred_df: pd.DataFrame, signal_name: str, filename: str):
    """Heatmap: sector x horizon, colored by t-stat of predictive coefficient."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    t_pivot = pred_df.pivot(index="sector", columns="horizon", values="t_b")
    b_pivot = pred_df.pivot(index="sector", columns="horizon", values="b")

    import seaborn as sns
    annot = b_pivot.copy().astype(object)
    for r in b_pivot.index:
        for c in b_pivot.columns:
            b = b_pivot.loc[r, c]
            t = t_pivot.loc[r, c]
            if np.isnan(b):
                annot.loc[r, c] = "N/A"
            else:
                star = "***" if abs(t) > 2.576 else "**" if abs(t) > 1.96 else "*" if abs(t) > 1.645 else ""
                annot.loc[r, c] = f"{b:.5f}\n(t={t:.2f}){star}"

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        t_pivot, ax=ax,
        cmap="RdYlGn", center=0,
        annot=annot, fmt="",
        linewidths=0.5, linecolor="#cccccc",
        cbar_kws={"label": "t-statistic of b"},
    )
    ax.set_title(
        f"Predictive Regression: {signal_name} -> Forward Returns\n"
        "cell = coefficient b  (t-stat colored)  |  * p<10%  ** p<5%  *** p<1%",
        fontsize=11, pad=10,
    )
    ax.set_xlabel("Horizon (days)", fontsize=10)
    ax.set_ylabel("Sector ETF", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def plot_portfolio(
    port_series_list: list[tuple[str, pd.Series]],
    spy_returns: pd.Series,
    filename: str = "longshort_portfolio.png",
):
    """Cumulative return of long-short portfolios vs SPY."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    ax1, ax2 = axes
    spy_cum = spy_returns.cumsum()
    spy_cum.plot(ax=ax1, color="black", linewidth=1.2, label="SPY (long only)", linestyle="--")

    for name, series in port_series_list:
        cum = series.cumsum()
        cum.plot(ax=ax1, linewidth=1.0, label=name, alpha=0.85)

    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_title("Long-Short Portfolio: Cumulative Log Return\n(Long top-3 / Short bottom-3 by signal, daily rebalance)", fontsize=10)
    ax1.set_ylabel("Cumulative log return")
    ax1.legend(fontsize=8)

    # Drawdown
    for name, series in port_series_list:
        cum = series.cumsum()
        roll_max = cum.cummax()
        dd = cum - roll_max
        dd.plot(ax=ax2, linewidth=0.9, label=name, alpha=0.8)

    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_title("Drawdown", fontsize=10)
    ax2.set_ylabel("Drawdown")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def print_portfolio_stats(name: str, series: pd.Series, spy: pd.Series):
    """Print CAGR, Sharpe, MDD for a daily return series."""
    ann    = series.mean() * 252
    vol    = series.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else np.nan
    cum    = series.cumsum()
    mdd    = (cum - cum.cummax()).min()
    print(f"  {name:30s}  CAGR={ann:.2%}  Vol={vol:.2%}  Sharpe={sharpe:.2f}  MDD={mdd:.2%}")


def print_is_oos_split(
    port_iv: pd.Series,
    port_vrp: pd.Series,
    spy_ret: pd.Series,
    oos_start: str,
):
    """Print IS vs OOS performance breakdown for both signals."""
    oos_dt = pd.Timestamp(oos_start)

    def _stats(s: pd.Series, label: str):
        if len(s) < 5:
            print(f"    {label:30s}  데이터 부족 ({len(s)}일)")
            return
        ann    = s.mean() * 252
        vol    = s.std() * np.sqrt(252)
        sharpe = ann / vol if vol > 0 else float("nan")
        cum    = s.cumsum()
        mdd    = (cum - cum.cummax()).min()
        n      = len(s)
        print(f"    {label:30s}  CAGR={ann:+.2%}  Vol={vol:.2%}  Sharpe={sharpe:+.2f}  MDD={mdd:.2%}  (n={n}일)")

    for sig_name, port in [("IV_spread", port_iv), ("VRP", port_vrp)]:
        is_  = port[port.index <  oos_dt]
        oos_ = port[port.index >= oos_dt]
        spy_is  = spy_ret[spy_ret.index <  oos_dt]
        spy_oos = spy_ret[spy_ret.index >= oos_dt]

        print(f"\n  [{sig_name} L/S]")
        print(f"  {'':->62}")
        print(f"  IN-SAMPLE  ({is_.index[0].date() if len(is_)>0 else '?'} ~ {(oos_dt - pd.Timedelta(days=1)).date()})")
        _stats(is_,  f"L/S {sig_name}")
        _stats(spy_is,  "SPY (long only)")
        print(f"  OUT-OF-SAMPLE  ({oos_dt.date()} ~ {port.index[-1].date() if len(oos_)>0 else '?'})")
        _stats(oos_, f"L/S {sig_name}")
        _stats(spy_oos, "SPY (long only)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sep = "=" * 62

    # Load cached data
    returns      = pd.read_parquet(OUTPUT_DIR / "sector_returns.parquet")
    macro_chg    = pd.read_parquet(OUTPUT_DIR / "macro_changes.parquet")
    macro_lvl    = pd.read_parquet(OUTPUT_DIR / "macro_levels.parquet")
    realized_vol = pd.read_parquet(OUTPUT_DIR / "realized_vol.parquet")
    capm_iv      = pd.read_parquet(OUTPUT_DIR / "capm_iv.parquet")

    sectors = [c for c in returns.columns if c != "SPY"]

    # Build signals
    iv_spread, vrp = build_iv_signals(capm_iv, realized_vol, macro_lvl)
    iv_spread.to_csv(OUTPUT_DIR / "iv_spread_series.csv")
    vrp.to_csv(OUTPUT_DIR / "vrp_capm_series.csv")

    # ── Part 1: Extended contemporaneous regression ──────────────────────────
    print(sep)
    print("PART 1 - Extended regression: macro + d(IV_spread)")
    print(sep)

    ext_df = run_extended_regression(returns, macro_chg, iv_spread)
    ext_df.to_csv(OUTPUT_DIR / "extended_regression.csv")

    print("\n  IV_spread delta and gamma (contemporaneous):")
    cols_iv = ["coef_IV_spread", "tval_IV_spread", "coef_IV_spread^2", "tval_IV_spread^2", "r2_adj"]
    print(ext_df[cols_iv].round(5).to_string())

    # R2 improvement over base model
    base_r2 = pd.read_csv(OUTPUT_DIR / "static_results.csv")
    base_r2_vix = base_r2[base_r2["macro"] == "VIX"].set_index("sector")["r2"]
    print("\n  R2_adj (extended model, all factors):")
    print(ext_df["r2_adj"].round(4).to_string())

    # ── Part 2: Predictive regression ────────────────────────────────────────
    print(f"\n{sep}")
    print("PART 2 - Predictive regression: signal(t) -> forward return(t+k)")
    print(sep)

    pred_iv  = run_predictive_regression(returns, iv_spread, signal_name="IV_spread")
    pred_vrp = run_predictive_regression(returns, vrp,       signal_name="VRP")

    pred_iv.to_csv(OUTPUT_DIR  / "predictive_iv_spread.csv",  index=False)
    pred_vrp.to_csv(OUTPUT_DIR / "predictive_vrp.csv",        index=False)

    print("\n  IV_spread predictive coefficients:")
    print(pred_iv[["sector", "horizon", "b", "t_b", "r2"]].to_string(index=False))

    print("\n  VRP predictive coefficients:")
    print(pred_vrp[["sector", "horizon", "b", "t_b", "r2"]].to_string(index=False))

    # Significant predictive signals
    print("\n  Significant predictors (|t_b| > 1.96):")
    sig = pd.concat([pred_iv, pred_vrp])
    sig = sig[sig["t_b"].abs() > 1.96].sort_values("t_b", key=abs, ascending=False)
    if sig.empty:
        print("  None at 5% level.")
    else:
        print(sig[["signal","sector","horizon","b","t_b","r2"]].to_string(index=False))

    plot_predictive_heatmap(pred_iv,  "IV_spread", "predictive_iv_spread.png")
    plot_predictive_heatmap(pred_vrp, "VRP",       "predictive_vrp.png")

    # ── Part 3: Long-short portfolio ─────────────────────────────────────────
    print(f"\n{sep}")
    print("PART 3 - Long-short portfolio backtest (top3 - bottom3)")
    print(sep)

    port_iv  = run_longshort_backtest(returns, iv_spread, signal_name="L/S IV_spread")
    port_vrp = run_longshort_backtest(returns, vrp,       signal_name="L/S VRP")

    spy_ret = returns["SPY"].dropna()

    print("\n  Performance summary:")
    print(f"  {'Signal':30s}  {'CAGR':>8}  {'Vol':>8}  {'Sharpe':>8}  {'MDD':>8}")
    print("  " + "-"*58)
    print_portfolio_stats("L/S IV_spread", port_iv,  spy_ret)
    print_portfolio_stats("L/S VRP",       port_vrp, spy_ret)
    print_portfolio_stats("SPY (long)",    spy_ret,  spy_ret)

    port_iv.to_csv(OUTPUT_DIR  / "portfolio_iv_spread.csv")
    port_vrp.to_csv(OUTPUT_DIR / "portfolio_vrp.csv")

    plot_portfolio(
        [("L/S IV_spread", port_iv), ("L/S VRP", port_vrp)],
        spy_ret,
        "longshort_portfolio.png",
    )

    # ── IS / OOS 분리 평가 ────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("PART 4 - IN-SAMPLE vs OUT-OF-SAMPLE breakdown")
    print(f"  Training : 2025-01-01 ~ 2025-12-31")
    print(f"  OOS Test : {OOS_START} ~ present")
    print(sep)
    print_is_oos_split(port_iv, port_vrp, spy_ret, OOS_START)

    # OOS 누적수익 단독 저장
    oos_dt = pd.Timestamp(OOS_START)
    for name, port in [("iv_spread", port_iv), ("vrp", port_vrp)]:
        oos_series = port[port.index >= oos_dt]
        oos_series.to_csv(OUTPUT_DIR / f"portfolio_{name}_oos.csv")

    print(f"\n  All outputs -> {OUTPUT_DIR}")
    print(f"  Charts      -> {FIGURES_DIR}")

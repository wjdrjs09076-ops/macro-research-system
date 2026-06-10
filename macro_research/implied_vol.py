"""
implied_vol.py - Implied Volatility estimation (three methods)

1. Market IV (snapshot)  : yfinance options chain -> Black-Scholes inversion (today only)
2. CAPM IV (historical)  : sqrt(beta^2 * VIX^2 + sigma_idio^2) — 2014~present
3. GARCH IV (historical) : GARCH(1,1) conditional vol — already in garch_vol.parquet

BS inversion uses Newton-Raphson on the call price formula:
    C = S*N(d1) - K*e^(-rT)*N(d2)
    d1 = (ln(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import scipy.stats as ss
import yfinance as yf

from config import SECTOR_ETFS, OUTPUT_DIR, FIGURES_DIR


# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

def _bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * ss.norm.cdf(d1) - K * np.exp(-r * T) * ss.norm.cdf(d2)


def _bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 1e-8
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * ss.norm.pdf(d1) * np.sqrt(T)


def bs_implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float = 0.05,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> float:
    """Newton-Raphson Black-Scholes implied volatility inversion."""
    sigma = 0.20  # initial guess
    for _ in range(max_iter):
        price = _bs_call_price(S, K, T, r, sigma)
        vega  = _bs_vega(S, K, T, r, sigma)
        diff  = market_price - price
        if abs(diff) < tol:
            break
        if vega < 1e-8:
            return np.nan
        sigma += diff / vega
        if sigma <= 0:
            return np.nan
    return sigma if 0 < sigma < 5 else np.nan


# ---------------------------------------------------------------------------
# Method 1: Market IV snapshot (today)
# ---------------------------------------------------------------------------

def get_market_iv_snapshot(
    tickers: list[str],
    target_days: int = 30,
    r: float = 0.05,
) -> pd.DataFrame:
    """
    Fetch current ATM call IV from yfinance options for each sector ETF.
    Returns DataFrame with columns: [ticker, spot, strike, expiry, days_to_exp, iv_market].
    """
    results = []

    for ticker in tickers:
        try:
            tk   = yf.Ticker(ticker)
            exps = tk.options
            if not exps:
                raise ValueError("no options available")

            # 만기 중 target_days에 가장 가까운 것 선택
            target_date = datetime.now() + timedelta(days=target_days)
            expiry = min(
                exps,
                key=lambda x: abs(datetime.strptime(x, "%Y-%m-%d") - target_date),
            )
            dte = (datetime.strptime(expiry, "%Y-%m-%d") - datetime.now()).days
            T   = dte / 365.0

            hist  = tk.history(period="2d")
            if hist.empty:
                raise ValueError("no price data")
            spot = float(hist["Close"].iloc[-1])

            chain = tk.option_chain(expiry)
            calls = chain.calls.dropna(subset=["impliedVolatility", "lastPrice"])
            calls = calls[calls["volume"] > 0] if "volume" in calls.columns else calls

            # ATM 콜 선택 (행사가가 현재가와 가장 가까운 것)
            atm_idx = (calls["strike"] - spot).abs().idxmin()
            atm     = calls.loc[atm_idx]
            strike  = float(atm["strike"])
            mkt_p   = float(atm["lastPrice"])

            # yfinance가 이미 IV를 제공하지만 BS로 직접 역산해 검증
            iv_yfin = float(atm["impliedVolatility"])
            iv_bs   = bs_implied_vol(mkt_p, spot, strike, T, r)

            results.append({
                "ticker":      ticker,
                "spot":        round(spot, 2),
                "strike":      strike,
                "expiry":      expiry,
                "days_to_exp": dte,
                "iv_yfinance": round(iv_yfin, 4),
                "iv_bs":       round(iv_bs, 4) if not np.isnan(iv_bs) else np.nan,
                "mkt_price":   round(mkt_p, 4),
            })
            print(f"  {ticker:5s}: spot={spot:.2f}  strike={strike:.2f}"
                  f"  DTE={dte}  IV(yf)={iv_yfin:.2%}  IV(BS)={iv_bs:.2%}" if not np.isnan(iv_bs) else
                  f"  {ticker:5s}: spot={spot:.2f}  IV(yf)={iv_yfin:.2%}")

        except Exception as e:
            print(f"  {ticker:5s}: ERROR - {e}")
            results.append({"ticker": ticker, **{k: np.nan for k in
                ["spot","strike","expiry","days_to_exp","iv_yfinance","iv_bs","mkt_price"]}})

    return pd.DataFrame(results).set_index("ticker")


# ---------------------------------------------------------------------------
# Method 2: CAPM-decomposed historical IV
# ---------------------------------------------------------------------------

def compute_capm_iv(
    returns: pd.DataFrame,
    vix_levels: pd.Series,
    window: int = 63,
) -> pd.DataFrame:
    """
    Historical sector IV via CAPM decomposition:

        IV_sector(t) = sqrt( beta(t)^2 * sigma_mkt(t)^2 + sigma_idio(t)^2 )

    where:
        beta(t)        = rolling 63D OLS beta of sector returns on SPY returns
        sigma_mkt(t)   = VIX(t) / 100  (market IV, annualized decimal)
        sigma_idio(t)  = sqrt( RV_sector^2 - beta^2 * RV_mkt^2 )
                         clipped at 0 to avoid sqrt of negative
    """
    if "SPY" not in returns.columns:
        raise ValueError("SPY must be in returns DataFrame")

    sectors   = [c for c in returns.columns if c != "SPY"]
    spy_ret   = returns["SPY"]

    # rolling beta
    betas: dict[str, pd.Series] = {}
    for sec in sectors:
        sec_ret = returns[sec]
        b_vals  = []
        dates   = []
        for end in range(window, len(returns) + 1):
            r_s = sec_ret.iloc[end - window:end].values
            r_m = spy_ret.iloc[end - window:end].values
            mask = ~(np.isnan(r_s) | np.isnan(r_m))
            if mask.sum() < 20:
                b_vals.append(np.nan)
            else:
                cov = np.cov(r_s[mask], r_m[mask])
                b_vals.append(cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else np.nan)
            dates.append(returns.index[end - 1])
        betas[sec] = pd.Series(b_vals, index=dates)

    beta_df = pd.DataFrame(betas)

    # rolling realized vol (annualized)
    rv = returns.rolling(window, min_periods=window // 2).std() * np.sqrt(252)

    # market IV from VIX level (already annualized %)
    # VIX is quoted as percentage (e.g. 20 = 20% annual vol)
    vix_aligned = vix_levels.reindex(returns.index).ffill() / 100.0  # decimal

    capm_iv = {}
    for sec in sectors:
        beta_s   = beta_df[sec].reindex(returns.index)
        rv_sec   = rv[sec]
        rv_mkt   = rv["SPY"]

        # idiosyncratic variance (clipped at 0)
        var_idio = (rv_sec ** 2 - beta_s ** 2 * rv_mkt ** 2).clip(lower=0)

        # CAPM IV
        iv_s = np.sqrt(beta_s ** 2 * vix_aligned ** 2 + var_idio)
        capm_iv[sec] = iv_s

    return pd.DataFrame(capm_iv)


# ---------------------------------------------------------------------------
# Comparison table and chart
# ---------------------------------------------------------------------------

def compare_iv_methods(
    capm_iv: pd.DataFrame,
    garch_iv: pd.DataFrame,
    market_iv_snap: pd.DataFrame,
) -> pd.DataFrame:
    """
    Latest-date snapshot comparing CAPM IV, GARCH IV, and Market IV (from options).
    """
    latest = capm_iv.dropna(how="all").index[-1]

    rows = []
    for sec in capm_iv.columns:
        row = {"sector": sec}
        row["capm_iv"]   = round(capm_iv.loc[latest, sec], 4) if sec in capm_iv.columns else np.nan
        row["garch_iv"]  = round(garch_iv.loc[latest, sec], 4) if (
            latest in garch_iv.index and sec in garch_iv.columns) else np.nan
        row["market_iv"] = round(market_iv_snap.loc[sec, "iv_bs"], 4) if sec in market_iv_snap.index else np.nan
        rows.append(row)

    return pd.DataFrame(rows).set_index("sector")


def plot_iv_comparison(
    capm_iv: pd.DataFrame,
    garch_iv: pd.DataFrame,
    filename: str = "iv_comparison.png",
):
    """CAPM IV vs GARCH IV time series for each sector."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sectors = capm_iv.columns.tolist()
    n = len(sectors)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.5 * rows))
    axes_flat = axes.flatten()

    for i, sec in enumerate(sectors):
        ax = axes_flat[i]
        if sec in capm_iv.columns:
            capm_iv[sec].rolling(21).mean().plot(
                ax=ax, label="CAPM IV (21D smooth)", color="steelblue", linewidth=1.0)
        if sec in garch_iv.columns:
            garch_iv[sec].rolling(21).mean().plot(
                ax=ax, label="GARCH IV (21D smooth)", color="darkorange",
                linewidth=1.0, linestyle="--")
        ax.set_title(f"{sec} ({SECTOR_ETFS.get(sec,'')})", fontsize=8)
        ax.set_ylabel("Ann. vol (decimal)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle("Sector IV: CAPM decomposition vs GARCH(1,1)  (21-day smoothed)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    tickers = list(SECTOR_ETFS.keys())

    # 캐시 로드
    returns   = pd.read_parquet(OUTPUT_DIR / "sector_returns.parquet")
    garch_iv  = pd.read_parquet(OUTPUT_DIR / "garch_vol.parquet")
    macro_lvl = pd.read_parquet(OUTPUT_DIR / "macro_levels.parquet")
    vix       = macro_lvl["VIX"]

    sep = "=" * 62

    # 1. Market IV snapshot
    print(sep)
    print("METHOD 1 - Market IV snapshot (BS inversion from options)")
    print(sep)
    market_iv_snap = get_market_iv_snapshot(tickers)
    market_iv_snap.to_csv(OUTPUT_DIR / "market_iv_snapshot.csv")
    print(f"\n  Saved: market_iv_snapshot.csv")

    # 2. CAPM IV history
    print(f"\n{sep}")
    print("METHOD 2 - CAPM-decomposed historical IV")
    print(sep)
    capm_iv = compute_capm_iv(returns, vix)
    capm_iv.to_parquet(OUTPUT_DIR / "capm_iv.parquet")
    capm_iv.to_csv(OUTPUT_DIR / "capm_iv_series.csv")
    print(f"  CAPM IV computed: {capm_iv.shape[0]} days x {capm_iv.shape[1]} sectors")

    # 3. Compare
    print(f"\n{sep}")
    print("IV COMPARISON - Latest snapshot (CAPM / GARCH / Market)")
    print(sep)
    comp = compare_iv_methods(capm_iv, garch_iv, market_iv_snap)
    comp.to_csv(OUTPUT_DIR / "iv_comparison.csv")
    print(comp.round(4).sort_values("market_iv", ascending=False).to_string())

    # 4. Chart
    plot_iv_comparison(capm_iv, garch_iv, "iv_comparison.png")

    # 5. VRP update with CAPM IV
    print(f"\n{sep}")
    print("VRP (CAPM-based vs GARCH-based)")
    print(sep)
    rv = pd.read_parquet(OUTPUT_DIR / "realized_vol.parquet")
    vrp_capm = capm_iv.sub(rv.reindex(capm_iv.index), fill_value=np.nan)
    vrp_capm.to_csv(OUTPUT_DIR / "vrp_capm_series.csv")
    print("\n  Latest VRP (CAPM IV - RV):")
    snap = vrp_capm.dropna(how="all").tail(1).T
    snap.columns = ["VRP_capm"]
    print(snap.sort_values("VRP_capm", ascending=False).round(4).to_string())

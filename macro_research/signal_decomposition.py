"""
signal_decomposition.py - IV_spread 신호 정보 함량 분석

IV_spread = CAPM_IV - VIX/100
         = sqrt(beta^2 * VIX^2 + sigma_idio^2) - VIX/100

핵심 질문:
  1. IV_spread를 beta-component vs sigma_idio-component로 분리할 때
     어느 쪽이 실제 예측력을 담당하는가?
  2. 2025 IS vs 2026 OOS에서 long/short 레그 구성이 어떻게 달랐나?
  3. VRP가 2026년에 망가진 원인 (RV spike? IV collapse? 구조 변화?)
  4. IV_spread의 롤링 예측력 — 시간이 지나도 안정적인가?
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
import statsmodels.api as sm
from scipy import stats

from config import OUTPUT_DIR, FIGURES_DIR, OOS_START, SECTOR_ETFS

OOS_DT = pd.Timestamp(OOS_START)
SECTORS = list(SECTOR_ETFS.keys())


# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------

def load_data():
    returns   = pd.read_parquet(OUTPUT_DIR / "sector_returns.parquet")
    capm_iv   = pd.read_parquet(OUTPUT_DIR / "capm_iv.parquet")
    garch_vol = pd.read_parquet(OUTPUT_DIR / "garch_vol.parquet")
    rv        = pd.read_parquet(OUTPUT_DIR / "realized_vol.parquet")
    macro_lvl = pd.read_parquet(OUTPUT_DIR / "macro_levels.parquet")

    iv_spread = pd.read_csv(OUTPUT_DIR / "iv_spread_series.csv", index_col=0, parse_dates=True)
    vrp       = pd.read_csv(OUTPUT_DIR / "vrp_capm_series.csv",  index_col=0, parse_dates=True)

    # SPY 제거
    rv_sec = rv[[c for c in rv.columns if c in SECTORS]]
    return returns, capm_iv, garch_vol, rv_sec, macro_lvl, iv_spread, vrp


# ---------------------------------------------------------------------------
# PART 1: IV_spread 성분 분해
# IV_spread = CAPM_IV - VIX/100
# CAPM_IV   = sqrt(beta^2 * VIX^2 + sigma_idio^2)
# 따라서:
#   beta_component = beta * VIX/100        (beta가 VIX에 기여하는 부분)
#   sigma_idio     = sqrt(CAPM_IV^2 - beta^2 * VIX^2)
#   IV_spread      = CAPM_IV - VIX/100
#                  ≈ (beta - 1) * VIX/100 + 추가 성분 (선형 근사)
# 두 신호를 직접 만들어서 각각의 예측력 비교
# ---------------------------------------------------------------------------

def decompose_iv_spread(capm_iv: pd.DataFrame, macro_lvl: pd.DataFrame, rv: pd.DataFrame):
    """
    Returns:
        beta_comp  : beta(t) * VIX(t)/100  per sector  (시장 베타가 만드는 IV)
        idio_comp  : sigma_idio(t)          per sector  (CAPM으로 설명 안 되는 고유 변동성)
        beta_spread: beta_comp - VIX/100    = (beta-1)*VIX/100  (beta 초과분)
    """
    vix = macro_lvl["VIX"].reindex(capm_iv.index).ffill() / 100.0

    # rolling 63D beta (capm_iv.parquet에는 저장 안 되어 있으므로 재계산)
    returns = pd.read_parquet(OUTPUT_DIR / "sector_returns.parquet")
    spy_ret = returns["SPY"]
    window  = 63

    beta_dict = {}
    for sec in SECTORS:
        if sec not in returns.columns:
            continue
        sec_ret = returns[sec]
        b_vals, dates = [], []
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
        beta_dict[sec] = pd.Series(b_vals, index=dates)

    beta_df = pd.DataFrame(beta_dict)

    vix_aligned  = vix.reindex(capm_iv.index).ffill()
    beta_aligned = beta_df.reindex(capm_iv.index)

    beta_comp   = beta_aligned.multiply(vix_aligned, axis=0)
    beta_spread = beta_comp.subtract(vix_aligned, axis=0)  # (beta-1)*VIX

    # sigma_idio = sqrt(max(CAPM_IV^2 - beta^2*VIX^2, 0))
    idio_var  = capm_iv ** 2 - beta_aligned ** 2 * vix_aligned.values.reshape(-1, 1) ** 2
    idio_comp = np.sqrt(idio_var.clip(lower=0))

    return beta_spread, idio_comp, beta_df


def predictive_reg_single(returns: pd.DataFrame, signal: pd.DataFrame,
                           horizon: int = 21, signal_name: str = "") -> pd.DataFrame:
    """sector별 signal(t) -> return(t+h) 회귀."""
    rows = []
    for sec in SECTORS:
        if sec not in returns.columns or sec not in signal.columns:
            continue
        fwd = returns[sec].rolling(horizon).sum().shift(-horizon)
        sig = signal[sec]
        common = sig.dropna().index.intersection(fwd.dropna().index)
        y, x   = fwd.loc[common].values, sig.loc[common].values
        mask   = ~(np.isnan(y) | np.isnan(x))
        if mask.sum() < 30:
            continue
        X = sm.add_constant(x[mask].reshape(-1, 1), has_constant="add")
        res = sm.OLS(y[mask], X).fit()
        rows.append({"sector": sec, "signal": signal_name,
                     "b": res.params[1], "t_b": res.tvalues[1], "r2": res.rsquared})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PART 2: Long/Short 레그 분석 (IS vs OOS)
# ---------------------------------------------------------------------------

def longshort_leg_analysis(returns: pd.DataFrame, signal: pd.DataFrame,
                            n_long: int = 3, n_short: int = 3) -> pd.DataFrame:
    """
    매일 long/short 레그 구성을 기록.
    Returns DataFrame: date x sector -> position (+1 long, -1 short, 0 none)
    """
    sectors = [c for c in returns.columns if c in signal.columns and c in SECTORS]
    common  = returns.index.intersection(signal.index)

    records = []
    for i in range(1, len(common)):
        date_prev = common[i - 1]
        date_today = common[i]
        sig_today = signal.loc[date_prev, sectors].dropna()
        if len(sig_today) < n_long + n_short:
            continue
        ranked = sig_today.sort_values(ascending=False)
        pos = {s: 0 for s in sectors}
        for s in ranked.iloc[:n_long].index:
            pos[s] = 1
        for s in ranked.iloc[-n_short:].index:
            pos[s] = -1
        records.append({"date": date_today, **pos})

    return pd.DataFrame(records).set_index("date")


def leg_frequency(positions: pd.DataFrame, period: str = "all") -> pd.DataFrame:
    """각 섹터가 long/short에 들어간 비율."""
    if period == "is":
        pos = positions[positions.index < OOS_DT]
    elif period == "oos":
        pos = positions[positions.index >= OOS_DT]
    else:
        pos = positions

    freq = pd.DataFrame({
        "long_pct":  (pos == 1).mean(),
        "short_pct": (pos == -1).mean(),
        "net_pct":   ((pos == 1).mean() - (pos == -1).mean()),
    })
    return freq.sort_values("net_pct", ascending=False)


# ---------------------------------------------------------------------------
# PART 3: VRP 구조 변화 분석
# ---------------------------------------------------------------------------

def vrp_structure_analysis(capm_iv: pd.DataFrame, rv: pd.DataFrame,
                            macro_lvl: pd.DataFrame) -> dict:
    """
    VRP = CAPM_IV - RV 의 IS vs OOS 통계 분석.
    - 평균, 분산, sign 비율 변화
    - RV spike 여부 (RV > CAPM_IV)
    - IV collapse 여부 (CAPM_IV 급락)
    """
    vix = macro_lvl["VIX"].reindex(capm_iv.index).ffill() / 100.0
    vrp_all = capm_iv.subtract(rv.reindex(capm_iv.index), fill_value=np.nan)

    results = {}
    for sec in SECTORS:
        if sec not in vrp_all.columns:
            continue
        v = vrp_all[sec].dropna()
        is_v  = v[v.index <  OOS_DT]
        oos_v = v[v.index >= OOS_DT]

        iv_s  = capm_iv[sec].dropna()
        rv_s  = rv[sec].reindex(capm_iv.index).dropna() if sec in rv.columns else pd.Series(dtype=float)

        results[sec] = {
            "vrp_mean_is":    is_v.mean(),
            "vrp_mean_oos":   oos_v.mean(),
            "vrp_std_is":     is_v.std(),
            "vrp_std_oos":    oos_v.std(),
            "pct_positive_is":  (is_v  > 0).mean(),
            "pct_positive_oos": (oos_v > 0).mean(),
            "rv_mean_is":  rv_s[rv_s.index <  OOS_DT].mean() if len(rv_s) > 0 else np.nan,
            "rv_mean_oos": rv_s[rv_s.index >= OOS_DT].mean() if len(rv_s) > 0 else np.nan,
            "iv_mean_is":  iv_s[iv_s.index <  OOS_DT].mean(),
            "iv_mean_oos": iv_s[iv_s.index >= OOS_DT].mean(),
        }

    return pd.DataFrame(results).T


# ---------------------------------------------------------------------------
# PART 4: 롤링 예측력 (60D 창으로 슬라이딩)
# ---------------------------------------------------------------------------

def rolling_predictive_power(returns: pd.DataFrame, signal: pd.DataFrame,
                               horizon: int = 21, window: int = 60) -> pd.DataFrame:
    """
    60일 창으로 슬라이딩하면서 신호의 평균 예측 t-stat 계산.
    반환: date -> mean_abs_t (11개 섹터 평균)
    """
    sectors = [c for c in SECTORS if c in returns.columns and c in signal.columns]
    common  = returns.index.intersection(signal.index)

    dates, mean_t = [], []
    for end in range(window + horizon, len(common)):
        window_dates = common[end - window:end]
        chunk_t = []
        for sec in sectors:
            fwd = returns[sec].rolling(horizon).sum().shift(-horizon)
            sig = signal[sec]
            y = fwd.loc[window_dates].values
            x = sig.loc[window_dates].values
            mask = ~(np.isnan(y) | np.isnan(x))
            if mask.sum() < 15:
                continue
            try:
                X   = sm.add_constant(x[mask].reshape(-1, 1), has_constant="add")
                res = sm.OLS(y[mask], X).fit()
                chunk_t.append(res.tvalues[1])
            except Exception:
                pass
        if chunk_t:
            dates.append(common[end])
            mean_t.append(np.mean(chunk_t))

    return pd.Series(mean_t, index=dates, name="mean_t_stat")


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------

def plot_all(beta_spread, idio_comp, iv_spread, vrp,
             vrp_struct, positions_iv, positions_vrp,
             rolling_t_iv, rolling_t_vrp):

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(20, 24))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)

    colors = plt.cm.tab20(np.linspace(0, 1, len(SECTORS)))
    sec_color = dict(zip(SECTORS, colors))

    # ── (0,0) beta_spread vs idio_comp 최신값 바 차트 ────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    latest_beta = beta_spread.dropna(how="all").iloc[-1].reindex(SECTORS)
    latest_idio = idio_comp.dropna(how="all").iloc[-1].reindex(SECTORS)
    x = np.arange(len(SECTORS))
    ax.bar(x - 0.2, latest_beta.values, 0.38, label="Beta spread (beta-1)*VIX",
           color="#1565C0", alpha=0.8)
    ax.bar(x + 0.2, latest_idio.values, 0.38, label="Idio vol (sigma_idio)",
           color="#C62828", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(SECTORS, rotation=30, fontsize=8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title("IV_spread 성분 분해 (최신)\nbeta-1)*VIX vs sigma_idio", fontsize=9)
    ax.legend(fontsize=7)
    ax.set_ylabel("Annualized vol", fontsize=8)

    # ── (0,1) beta_spread vs idio_comp 예측력 비교 ───────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    # 미리 계산된 t-stats (메인 함수에서 전달받음)
    # 여기서는 placeholder — 실제값은 아래 main에서 직접 출력
    ax.text(0.5, 0.5, "See console output\nfor component predictive t-stats",
            ha="center", va="center", transform=ax.transAxes, fontsize=11,
            color="gray")
    ax.set_title("성분별 예측력 (t-stat)\n→ 콘솔 출력 참조", fontsize=9)
    ax.axis("off")

    # ── (1,0) IS vs OOS long/short 레그 빈도 (IV_spread) ────────────────────
    ax = fig.add_subplot(gs[1, 0])
    freq_is  = leg_frequency(positions_iv, "is")
    freq_oos = leg_frequency(positions_iv, "oos")
    x = np.arange(len(freq_is))
    ax.bar(x - 0.2, freq_is["net_pct"].values,  0.38,
           label="IS net (long-short freq)", color="#1565C0", alpha=0.8)
    ax.bar(x + 0.2, freq_oos["net_pct"].reindex(freq_is.index).values, 0.38,
           label="OOS net", color="#E65100", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(freq_is.index, rotation=30, fontsize=8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title("IV_spread L/S: 섹터별 롱/숏 빈도 차이\n(양수=자주 롱, 음수=자주 숏)", fontsize=9)
    ax.legend(fontsize=7)

    # ── (1,1) VRP 구조 변화 — IS vs OOS 평균 VRP ─────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    vrp_is  = vrp_struct["vrp_mean_is"].reindex(SECTORS)
    vrp_oos = vrp_struct["vrp_mean_oos"].reindex(SECTORS)
    x = np.arange(len(SECTORS))
    ax.bar(x - 0.2, vrp_is.values,  0.38,
           label="IS mean VRP (2025)", color="#1565C0", alpha=0.8)
    ax.bar(x + 0.2, vrp_oos.values, 0.38,
           label="OOS mean VRP (2026)", color="#C62828", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(SECTORS, rotation=30, fontsize=8)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_title("VRP 구조 변화: IS(2025) vs OOS(2026)\n0 아래 = IV < RV (변동성 폭발)", fontsize=9)
    ax.legend(fontsize=7)
    ax.set_ylabel("VRP (연환산)", fontsize=8)

    # ── (2,0) CAPM IV vs RV 시계열 (대표 섹터 XLI, XLB) ─────────────────────
    ax = fig.add_subplot(gs[2, 0])
    capm_iv = pd.read_parquet(OUTPUT_DIR / "capm_iv.parquet")
    rv_df   = pd.read_parquet(OUTPUT_DIR / "realized_vol.parquet")
    for sec, ls in [("XLI", "-"), ("XLB", "--")]:
        if sec in capm_iv.columns:
            capm_iv[sec].rolling(5).mean().plot(ax=ax, label=f"{sec} CAPM IV",
                                                 linewidth=1.0, linestyle=ls)
        if sec in rv_df.columns:
            rv_df[sec].reindex(capm_iv.index).rolling(5).mean().plot(
                ax=ax, label=f"{sec} RV", linewidth=1.0, linestyle=ls, alpha=0.6)
    ax.axvline(OOS_DT, color="red", linewidth=1.2, linestyle=":", label="OOS start")
    ax.set_title("CAPM IV vs RV 시계열 (XLI, XLB)\nRV > IV 구간 = VRP < 0 (VRP 신호 붕괴)", fontsize=9)
    ax.legend(fontsize=6.5, ncol=2)
    ax.set_ylabel("Annualized vol", fontsize=8)
    ax.tick_params(labelsize=7)

    # ── (2,1) IV_spread 시계열 (전체) ────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    for sec in SECTORS:
        if sec in iv_spread.columns:
            iv_spread[sec].rolling(5).mean().plot(ax=ax, label=sec,
                                                   linewidth=0.8, alpha=0.75)
    ax.axvline(OOS_DT, color="red", linewidth=1.2, linestyle=":", label="OOS start")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("IV_spread 시계열 (5D MA)\n양수 = 섹터 IV > 시장 beta-implied IV", fontsize=9)
    ax.legend(fontsize=5.5, ncol=3)
    ax.set_ylabel("IV_spread", fontsize=8)
    ax.tick_params(labelsize=7)

    # ── (3,0) 롤링 예측력 시계열 (IV_spread vs VRP) ───────────────────────────
    ax = fig.add_subplot(gs[3, 0])
    rolling_t_iv.plot(ax=ax,  label="IV_spread mean t-stat", color="#1565C0", linewidth=1.2)
    rolling_t_vrp.plot(ax=ax, label="VRP mean t-stat",       color="#C62828", linewidth=1.2,
                       linestyle="--")
    ax.axhline(0,    color="black", linewidth=0.5)
    ax.axhline(1.96, color="green", linewidth=0.8, linestyle=":", alpha=0.6, label="+1.96")
    ax.axhline(-1.96,color="red",   linewidth=0.8, linestyle=":", alpha=0.6, label="-1.96")
    ax.axvline(OOS_DT, color="red", linewidth=1.2, linestyle=":", label="OOS start")
    ax.set_title("롤링 예측력 (60D 창, 21D 선행)\n평균 t-stat across 11 sectors", fontsize=9)
    ax.legend(fontsize=7)
    ax.set_ylabel("mean t-stat", fontsize=8)
    ax.tick_params(labelsize=7)

    # ── (3,1) IS vs OOS VRP 양수 비율 ────────────────────────────────────────
    ax = fig.add_subplot(gs[3, 1])
    pct_is  = vrp_struct["pct_positive_is"].reindex(SECTORS)
    pct_oos = vrp_struct["pct_positive_oos"].reindex(SECTORS)
    x = np.arange(len(SECTORS))
    ax.bar(x - 0.2, pct_is.values,  0.38,
           label="IS: VRP>0 비율", color="#1565C0", alpha=0.8)
    ax.bar(x + 0.2, pct_oos.values, 0.38,
           label="OOS: VRP>0 비율", color="#C62828", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(SECTORS, rotation=30, fontsize=8)
    ax.axhline(0.5, color="black", linewidth=0.7, linestyle="--", label="50%")
    ax.set_title("VRP>0 (IV>RV) 비율: IS vs OOS\n낮을수록 변동성이 실제로 더 크게 터짐", fontsize=9)
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1)

    fig.suptitle("IV_spread 신호 정보 함량 분석  |  IS(2025) vs OOS(2026)",
                 fontsize=13, y=1.01)
    out = FIGURES_DIR / "signal_decomposition.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sep = "=" * 62

    print(f"\n{sep}")
    print("  IV_spread 신호 정보 함량 분석")
    print(sep)

    returns, capm_iv, garch_vol, rv, macro_lvl, iv_spread, vrp = load_data()

    # ── PART 1: 성분 분해 ─────────────────────────────────────────────────────
    print(f"\n[1] IV_spread 성분 분해 계산 중...")
    beta_spread, idio_comp, beta_df = decompose_iv_spread(capm_iv, macro_lvl, rv)

    # 각 성분의 21D 예측력
    pred_full    = predictive_reg_single(returns, iv_spread,   21, "IV_spread(full)")
    pred_beta    = predictive_reg_single(returns, beta_spread, 21, "beta_spread")
    pred_idio    = predictive_reg_single(returns, idio_comp,   21, "idio_vol")

    print(f"\n  [성분별 21D 예측 t-stat 비교]")
    print(f"  {'Sector':6s}  {'IV_spread':>12}  {'beta_spread':>12}  {'idio_vol':>12}")
    print("  " + "-" * 48)
    for sec in SECTORS:
        def _t(df, s):
            row = df[df["sector"] == s]
            return f"{row['t_b'].values[0]:+.2f}" if len(row) > 0 else "  N/A"
        print(f"  {sec:6s}  {_t(pred_full, sec):>12}  {_t(pred_beta, sec):>12}  {_t(pred_idio, sec):>12}")

    # 어느 성분이 더 강한지 집계
    t_full = pred_full.set_index("sector")["t_b"].reindex(SECTORS)
    t_beta = pred_beta.set_index("sector")["t_b"].reindex(SECTORS)
    t_idio = pred_idio.set_index("sector")["t_b"].reindex(SECTORS)
    print(f"\n  Mean |t|:  IV_spread={t_full.abs().mean():.2f}  "
          f"beta_spread={t_beta.abs().mean():.2f}  idio_vol={t_idio.abs().mean():.2f}")
    winner = max(
        [("IV_spread", t_full.abs().mean()),
         ("beta_spread", t_beta.abs().mean()),
         ("idio_vol", t_idio.abs().mean())],
        key=lambda x: x[1]
    )[0]
    print(f"  >> 예측력 우위: {winner}")

    # IS vs OOS 성분 예측력
    print(f"\n  [IS vs OOS 성분 예측력]")
    for period, mask in [("IS (2025)", returns.index < OOS_DT),
                          ("OOS (2026)", returns.index >= OOS_DT)]:
        ret_p = returns[mask]
        iv_p  = iv_spread.reindex(ret_p.index)
        bs_p  = beta_spread.reindex(ret_p.index)
        id_p  = idio_comp.reindex(ret_p.index)
        if len(ret_p) < 30:
            continue
        p_full = predictive_reg_single(ret_p, iv_p,  21, "IV_spread")
        p_beta = predictive_reg_single(ret_p, bs_p,  21, "beta_spread")
        p_idio = predictive_reg_single(ret_p, id_p,  21, "idio_vol")
        tf = p_full["t_b"].abs().mean() if len(p_full) > 0 else float("nan")
        tb = p_beta["t_b"].abs().mean() if len(p_beta) > 0 else float("nan")
        ti = p_idio["t_b"].abs().mean() if len(p_idio) > 0 else float("nan")
        print(f"  {period:12s}  mean|t|: IV_spread={tf:.2f}  beta_spread={tb:.2f}  idio_vol={ti:.2f}")

    # ── PART 2: Long/Short 레그 분석 ─────────────────────────────────────────
    print(f"\n{sep}")
    print("[2] Long/Short 레그 구성 분석")
    print(sep)

    positions_iv  = longshort_leg_analysis(returns, iv_spread)
    positions_vrp = longshort_leg_analysis(returns, vrp)

    for sig_name, pos in [("IV_spread", positions_iv), ("VRP", positions_vrp)]:
        print(f"\n  [{sig_name}]  IS vs OOS 섹터별 포지션 빈도")
        fi = leg_frequency(pos, "is")
        fo = leg_frequency(pos, "oos")
        print(f"  {'Sector':6s}  {'IS long%':>8}  {'IS short%':>9}  "
              f"{'OOS long%':>9}  {'OOS short%':>10}  {'방향 유지':>8}")
        print("  " + "-" * 60)
        for sec in SECTORS:
            il = fi.loc[sec, "long_pct"]  if sec in fi.index else 0
            is_ = fi.loc[sec, "short_pct"] if sec in fi.index else 0
            ol = fo.loc[sec, "long_pct"]  if sec in fo.index else 0
            os_ = fo.loc[sec, "short_pct"] if sec in fo.index else 0
            # 방향 유지 여부: IS net과 OOS net 같은 부호?
            is_net = il - is_
            oos_net = ol - os_
            consistent = "O" if (is_net * oos_net > 0) else ("X" if (is_net * oos_net < 0) else "-")
            print(f"  {sec:6s}  {il:>8.1%}  {is_:>9.1%}  {ol:>9.1%}  {os_:>10.1%}  {consistent:>8}")

    # ── PART 3: VRP 구조 변화 ─────────────────────────────────────────────────
    print(f"\n{sep}")
    print("[3] VRP 구조 변화 분석 (IS vs OOS)")
    print(sep)

    vrp_struct = vrp_structure_analysis(capm_iv, rv, macro_lvl)
    print(f"\n  {'Sector':6s}  {'VRP IS':>8}  {'VRP OOS':>8}  {'변화':>8}  "
          f"{'RV IS':>7}  {'RV OOS':>7}  {'IV IS':>7}  {'IV OOS':>7}  {'VRP>0 IS':>9}  {'VRP>0 OOS':>10}")
    print("  " + "-" * 90)
    for sec in SECTORS:
        if sec not in vrp_struct.index:
            continue
        r = vrp_struct.loc[sec]
        delta = r["vrp_mean_oos"] - r["vrp_mean_is"]
        print(f"  {sec:6s}  {r['vrp_mean_is']:>8.4f}  {r['vrp_mean_oos']:>8.4f}  "
              f"{delta:>+8.4f}  {r['rv_mean_is']:>7.4f}  {r['rv_mean_oos']:>7.4f}  "
              f"{r['iv_mean_is']:>7.4f}  {r['iv_mean_oos']:>7.4f}  "
              f"{r['pct_positive_is']:>9.1%}  {r['pct_positive_oos']:>10.1%}")

    # VRP 실패 원인 진단
    rv_change = vrp_struct["rv_mean_oos"] - vrp_struct["rv_mean_is"]
    iv_change = vrp_struct["iv_mean_oos"] - vrp_struct["iv_mean_is"]
    print(f"\n  [VRP 붕괴 원인 진단]")
    print(f"  평균 RV 변화 (IS->OOS): {rv_change.mean():+.4f}  "
          f"({'상승' if rv_change.mean() > 0 else '하락'})")
    print(f"  평균 IV 변화 (IS->OOS): {iv_change.mean():+.4f}  "
          f"({'상승' if iv_change.mean() > 0 else '하락'})")
    dominant = "RV 급등" if rv_change.mean() > abs(iv_change.mean()) else "IV 붕괴"
    print(f"  >> 주요 원인: {dominant}")
    print(f"  >> VRP>0 비율: IS 평균 {vrp_struct['pct_positive_is'].mean():.1%}  "
          f"-> OOS {vrp_struct['pct_positive_oos'].mean():.1%}")

    # ── PART 4: 롤링 예측력 ───────────────────────────────────────────────────
    print(f"\n{sep}")
    print("[4] 롤링 예측력 시계열 (60D 창, 21D 선행)")
    print(sep)

    rolling_t_iv  = rolling_predictive_power(returns, iv_spread, horizon=21, window=60)
    rolling_t_vrp = rolling_predictive_power(returns, vrp,       horizon=21, window=60)

    oos_t_iv  = rolling_t_iv[rolling_t_iv.index  >= OOS_DT]
    oos_t_vrp = rolling_t_vrp[rolling_t_vrp.index >= OOS_DT]
    is_t_iv   = rolling_t_iv[rolling_t_iv.index  <  OOS_DT]
    is_t_vrp  = rolling_t_vrp[rolling_t_vrp.index < OOS_DT]

    print(f"\n  IV_spread  IS mean_t={is_t_iv.mean():+.2f}   OOS mean_t={oos_t_iv.mean():+.2f}")
    print(f"  VRP        IS mean_t={is_t_vrp.mean():+.2f}   OOS mean_t={oos_t_vrp.mean():+.2f}")
    print(f"\n  IV_spread OOS 안정성: {('유지' if oos_t_iv.mean() * is_t_iv.mean() > 0 else '반전')}")
    print(f"  VRP       OOS 안정성: {('유지' if len(oos_t_vrp)>0 and oos_t_vrp.mean() * is_t_vrp.mean() > 0 else '반전')}")

    # ── 시각화 ────────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("[5] 시각화 생성...")
    plot_all(beta_spread, idio_comp, iv_spread, vrp,
             vrp_struct, positions_iv, positions_vrp,
             rolling_t_iv, rolling_t_vrp)

    print(f"\n{sep}")
    print("  분석 완료.")
    print(sep)

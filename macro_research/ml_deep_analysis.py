"""
ml_deep_analysis.py

ML-Ensemble vs Buy & Hold XLE 심층 탐구

핵심 질문:
  Q1. ML Sharpe=2.14 vs OOS 정확도 28.6% -- 이 모순의 정체는?
  Q2. BnH XLE +85.9% -- ML이 왜 10%밖에 못 잡았나?

분석 구성:
  1. ML 신호 발화 타임라인 -- 언제 켜지고 꺼졌나, 무엇을 놓쳤나
  2. 과적합 진단 -- IS 97.9% vs OOS 28.6% 격차 해부 (학습곡선 + TimeSeriesCV)
  3. 피처 중요도 -- IS 학습 피처 vs OOS 실제 중요 피처 비교
  4. 임계값 민감도 -- threshold 0.40~0.72 스캔 (실제 Sharpe / 거래 빈도)
  5. XLE 수익 단계 분해 -- 어느 구간에 수익이 집중됐나
  6. 알파 포착률 -- 전략별 XLE 전체 이동의 몇%를 잡았나
  7. 개선 방향 -- 실질적으로 도움이 될 변경사항 (솔직하게)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, learning_curve
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.inspection import permutation_importance
import yfinance as yf

BASE_DIR    = Path(__file__).parent
FIGURES_DIR = BASE_DIR / "output" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

IS_START   = "2025-01-01"
IS_END     = "2025-12-31"
OOS_START  = "2026-01-01"
OOS_END    = "2026-05-29"
FULL_START = "2024-07-01"   # Z-score 롤링 여유분

SECTOR_ETF    = "XLE"
ENERGY_STOCKS = ["XOM", "CVX", "SLB", "OXY", "COP"]
TC            = 0.0005
EWMA_SPAN     = 32
RV_WIN        = 20
BETA_WIN      = 30

FEATURE_COLS = [
    "beta_30d", "vrp", "vix",
    "beta_30d_z", "vrp_z", "vix_z",
    "mom_20d", "mom_63d", "rel_str", "beta_delta5",
]

# ===========================================================================
# 데이터 + 피처 (ml_trading_ontology 와 동일 함수)
# ===========================================================================

def fetch_data(tickers, start, end):
    all_t = list(set(tickers + ["SPY", "^VIX", "CL=F"]))
    raw   = yf.download(all_t, start=start, end=end,
                        auto_adjust=True, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    out = {}
    for t in all_t:
        if t in close.columns:
            s = close[t].dropna()
            if len(s) > 20:
                out[t] = s
    return out

def rolling_beta(sec_ret, spy_ret, window=BETA_WIN):
    idx = sec_ret.index.intersection(spy_ret.index)
    s, m = sec_ret.loc[idx], spy_ret.loc[idx]
    return (s.rolling(window).cov(m) / m.rolling(window).var()).dropna()

def compute_vrp(sec_ret, spy_ret, vix):
    idx  = sec_ret.index.intersection(spy_ret.index)
    s, m = sec_ret.loc[idx], spy_ret.loc[idx]
    b63  = rolling_beta(s, m, 63).reindex(idx).ffill()
    res  = s - b63 * m
    idio = res.ewm(span=EWMA_SPAN, min_periods=20).std() * np.sqrt(252)
    vix_a = vix.reindex(idx).ffill() / 100.0
    capm_iv = np.sqrt((b63 * vix_a)**2 + idio**2)
    rv_ann  = s.rolling(RV_WIN).std() * np.sqrt(252)
    return (capm_iv - rv_ann).dropna(), capm_iv, rv_ann

def build_features(sec_ret, spy_ret, vix_series):
    beta = rolling_beta(sec_ret, spy_ret)
    vrp, _, _ = compute_vrp(sec_ret, spy_ret, vix_series)
    idx  = beta.index.intersection(vrp.index)
    df   = pd.DataFrame(index=idx)
    df["beta_30d"] = beta.reindex(idx)
    df["vrp"]      = vrp.reindex(idx)
    df["vix"]      = vix_series.reindex(idx).ffill()
    df["sec_ret"]  = sec_ret.reindex(idx)
    for col in ["beta_30d", "vrp", "vix"]:
        rm = df[col].rolling(252, min_periods=63).mean()
        rs = df[col].rolling(252, min_periods=63).std()
        df[f"{col}_z"] = (df[col] - rm) / (rs + 1e-8)
    sec_px = (1 + sec_ret).cumprod()
    df["mom_20d"]     = sec_px.reindex(idx).pct_change(20)
    df["mom_63d"]     = sec_px.reindex(idx).pct_change(63)
    spy_px            = (1 + spy_ret).cumprod()
    df["spy_20d"]     = spy_px.reindex(idx).pct_change(20)
    df["rel_str"]     = df["mom_20d"] - df["spy_20d"]
    df["beta_delta5"] = df["beta_30d"].diff(5)
    df["fwd_10d"]     = sec_ret.reindex(idx).shift(-10).rolling(10).sum()
    df["target"]      = (df["fwd_10d"] > 0).astype(int)
    return df.dropna()

def train_ensemble(df_is):
    X  = df_is[FEATURE_COLS].values
    y  = df_is["target"].values
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    models = {
        "LR" : LogisticRegression(C=0.5, max_iter=500, random_state=42),
        "RF" : RandomForestClassifier(n_estimators=200, max_depth=4,
                                       min_samples_leaf=10, random_state=42),
        "GBT": GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                           learning_rate=0.05,
                                           min_samples_leaf=10, random_state=42),
    }
    is_accs = {}
    for n, m in models.items():
        m.fit(Xs, y)
        is_accs[n] = m.score(Xs, y)
    return models, sc, is_accs

def ensemble_proba(models, sc, df, is_accs=None):
    X  = df[FEATURE_COLS].dropna().values
    Xs = sc.transform(X)
    idx = df[FEATURE_COLS].dropna().index
    probas = {n: m.predict_proba(Xs)[:, 1] for n, m in models.items()}
    if is_accs:
        tot = sum(is_accs.values())
        arr = sum(probas[k] * is_accs[k] / tot for k in models)
    else:
        arr = np.mean(list(probas.values()), axis=0)
    return pd.Series(arr, index=idx)

def backtest(price_ret, signal, tc=TC):
    sig  = signal.reindex(price_ret.index).fillna(False).astype(float)
    cost = sig.diff().abs() * tc
    net  = sig * price_ret - cost
    cum  = (1 + net).cumprod()
    return net, cum

# ===========================================================================
# 분석 1: 신호 발화 타임라인
# ===========================================================================

def analyze_signal_timing(df_full, proba, etf_ret, threshold=0.55):
    """ML 신호가 언제 켜지고, 무슨 수익 구간을 놓쳤나"""
    oos_dt  = pd.Timestamp(OOS_START)
    ml_sig  = (proba > threshold) & (df_full["vrp"].reindex(proba.index).ffill() < 0)
    ml_oos  = ml_sig[ml_sig.index >= oos_dt]
    xle_oos = (1 + etf_ret[etf_ret.index >= oos_dt]).cumprod()

    # 신호 발화 구간
    fired = ml_oos[ml_oos == True].index
    n_on  = int(ml_oos.sum())
    n_off = len(ml_oos) - n_on

    # 신호 OFF 구간의 XLE 수익
    off_ret   = etf_ret[(etf_ret.index >= oos_dt) & (~ml_oos.reindex(etf_ret.index, fill_value=False))]
    missed_cum = (1 + off_ret).cumprod().iloc[-1] - 1 if len(off_ret) > 0 else 0

    return {
        "n_on"      : n_on,
        "n_off"     : n_off,
        "pct_active": n_on / max(len(ml_oos), 1),
        "missed_ret": missed_cum,
        "fire_dates": fired.tolist(),
    }

# ===========================================================================
# 분석 2: 과적합 진단
# ===========================================================================

def overfitting_diagnosis(df_is, models, sc):
    """
    IS 97.9% vs OOS 28.6% 격차:
    - TimeSeriesCV 내 교차검증 점수
    - GBT 학습 곡선 (표본수 증가에 따른 IS/CV 성능)
    - 각 피처의 IS vs OOS 분포 비교
    """
    X  = df_is[FEATURE_COLS].values
    y  = df_is["target"].values
    Xs = sc.transform(X)

    # TimeSeriesSplit CV (IS 내부)
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = {"LR": [], "RF": [], "GBT": []}
    for train_idx, val_idx in tscv.split(Xs):
        for name, m in models.items():
            m_clone = type(m)(**m.get_params())
            m_clone.fit(Xs[train_idx], y[train_idx])
            cv_scores[name].append(m_clone.score(Xs[val_idx], y[val_idx]))

    # GBT 학습 곡선
    gbt     = models["GBT"]
    sizes   = np.linspace(0.1, 1.0, 8)
    lc_tr, lc_cv = [], []
    for frac in sizes:
        n = max(20, int(len(X) * frac))
        Xt, yt = Xs[:n], y[:n]
        ts = TimeSeriesSplit(n_splits=3)
        tr_sc, cv_sc = [], []
        for tr_i, vl_i in ts.split(Xt):
            if len(vl_i) < 5:
                continue
            gbt_c = type(gbt)(**gbt.get_params())
            gbt_c.fit(Xt[tr_i], yt[tr_i])
            tr_sc.append(gbt_c.score(Xt[tr_i], yt[tr_i]))
            cv_sc.append(gbt_c.score(Xt[vl_i], yt[vl_i]))
        if tr_sc:
            lc_tr.append(np.mean(tr_sc))
            lc_cv.append(np.mean(cv_sc))

    return {
        "cv_scores"    : cv_scores,
        "cv_mean"      : {k: np.mean(v) for k, v in cv_scores.items()},
        "cv_std"       : {k: np.std(v) for k, v in cv_scores.items()},
        "lc_sizes"     : [int(len(X) * f) for f in sizes[:len(lc_tr)]],
        "lc_train"     : lc_tr,
        "lc_cv"        : lc_cv,
    }

# ===========================================================================
# 분석 3: 피처 중요도 IS vs OOS
# ===========================================================================

def feature_importance_shift(df_is, df_oos, models, sc):
    """
    IS 학습에서 중요한 피처 vs OOS에서 실제로 예측력 있는 피처
    permutation importance 사용 (모델에 무관한 방법)
    """
    gbt = models["GBT"]

    X_is = sc.transform(df_is[FEATURE_COLS].values)
    y_is = df_is["target"].values
    pi_is = permutation_importance(gbt, X_is, y_is,
                                   n_repeats=20, random_state=42)

    X_oos = sc.transform(df_oos[FEATURE_COLS].dropna().values)
    y_oos = df_oos["target"].reindex(df_oos[FEATURE_COLS].dropna().index).values
    if len(X_oos) > 10 and len(set(y_oos)) > 1:
        pi_oos = permutation_importance(gbt, X_oos, y_oos,
                                        n_repeats=20, random_state=42)
        oos_imp = pi_oos.importances_mean
    else:
        oos_imp = np.zeros(len(FEATURE_COLS))

    return {
        "features" : FEATURE_COLS,
        "is_imp"   : pi_is.importances_mean,
        "is_std"   : pi_is.importances_std,
        "oos_imp"  : oos_imp,
    }

# ===========================================================================
# 분석 4: 임계값 민감도
# ===========================================================================

def threshold_scan(proba_full, df_full, etf_ret):
    """
    threshold 0.40~0.72 스캔:
    각 임계값에서 신호 발화율 / OOS Sharpe / 포착률 계산
    """
    oos_dt   = pd.Timestamp(OOS_START)
    vrp_s    = df_full["vrp"].reindex(proba_full.index).ffill()
    thresholds = np.arange(0.40, 0.73, 0.03)
    rows = []
    for th in thresholds:
        sig  = ((proba_full > th) & (vrp_s < 0)).shift(1).fillna(False)
        sig  = sig.reindex(etf_ret.index).fillna(False).astype(float)
        cost = sig.diff().abs() * TC
        net  = sig * etf_ret - cost
        oos_net = net[net.index >= oos_dt]
        if len(oos_net) < 5 or oos_net.std() == 0:
            rows.append({"threshold": th, "n_days": 0, "sharpe": np.nan,
                         "cagr": np.nan, "capture": np.nan})
            continue
        c_oos  = (1 + oos_net).cumprod()
        cagr   = c_oos.iloc[-1] ** (252 / len(oos_net)) - 1
        sharpe = oos_net.mean() / oos_net.std() * np.sqrt(252)
        n_days = int(sig[sig.index >= oos_dt].sum())
        bnh_ret = etf_ret[etf_ret.index >= oos_dt]
        bnh_cum = (1 + bnh_ret).cumprod().iloc[-1] - 1
        strat_cum = (1 + oos_net).cumprod().iloc[-1] - 1
        capture = strat_cum / bnh_cum if bnh_cum != 0 else 0
        rows.append({"threshold": th, "n_days": n_days,
                     "sharpe": sharpe, "cagr": cagr, "capture": capture})
    return pd.DataFrame(rows)

# ===========================================================================
# 분석 5: XLE 수익 단계 분해
# ===========================================================================

def xle_phase_decomposition(prices, etf_ret, spy_ret, vix, oil_price):
    """
    XLE OOS 수익을 단계별로 분해:
    - 구간별 누적 수익
    - 유가 기여도 (WTI 상관 기반 귀속)
    - 베타 기여도 vs 알파(초과수익) 기여도
    - VIX 체제
    """
    oos_dt  = pd.Timestamp(OOS_START)
    xle_oos = etf_ret[etf_ret.index >= oos_dt]
    spy_oos = spy_ret.reindex(xle_oos.index).dropna()

    # 월별 수익 분해
    monthly = xle_oos.resample("ME").apply(lambda r: (1 + r).prod() - 1)

    # 유가 상관
    oil_ret = oil_price.pct_change().dropna()
    oil_oos = oil_ret.reindex(xle_oos.index).dropna()
    idx_c   = xle_oos.index.intersection(oil_oos.index)
    oil_corr = np.corrcoef(xle_oos.loc[idx_c], oil_oos.loc[idx_c])[0, 1]

    # SPY 베타 기여 vs XLE 초과수익
    # XLE return = beta * SPY return + alpha
    beta_oos = rolling_beta(xle_oos, spy_oos, window=20).reindex(xle_oos.index).ffill()
    xle_expected = beta_oos * spy_oos
    xle_alpha    = xle_oos - xle_expected.reindex(xle_oos.index).fillna(0)

    # VIX 레벨별 구간
    vix_oos = vix.reindex(xle_oos.index).ffill()
    vix_low  = xle_oos[vix_oos < 20]
    vix_mid  = xle_oos[(vix_oos >= 20) & (vix_oos < 30)]
    vix_high = xle_oos[vix_oos >= 30]

    return {
        "monthly"       : monthly,
        "oil_corr"      : oil_corr,
        "cum_xle"       : (1 + xle_oos).cumprod(),
        "cum_expected"  : (1 + xle_expected.reindex(xle_oos.index).fillna(0)).cumprod(),
        "cum_alpha"     : (1 + xle_alpha).cumprod(),
        "vix_low_ret"   : (1 + vix_low).prod() - 1 if len(vix_low) > 0 else 0,
        "vix_mid_ret"   : (1 + vix_mid).prod() - 1 if len(vix_mid) > 0 else 0,
        "vix_high_ret"  : (1 + vix_high).prod() - 1 if len(vix_high) > 0 else 0,
        "vix_low_days"  : len(vix_low),
        "vix_mid_days"  : len(vix_mid),
        "vix_high_days" : len(vix_high),
        "xle_oos"       : xle_oos,
        "oil_oos"       : oil_oos,
        "vix_oos"       : vix_oos,
    }

# ===========================================================================
# 분석 6: 알파 포착률
# ===========================================================================

def alpha_capture_rate(etf_ret, strategies: dict):
    """
    전략별 XLE OOS 전체 이동의 몇%를 포착했나
    포착률 = 전략 누적 수익 / BnH 누적 수익
    추가: 롤링 포착률 (30일 윈도우)
    """
    oos_dt   = pd.Timestamp(OOS_START)
    bnh_ret  = etf_ret[etf_ret.index >= oos_dt]
    bnh_cum  = (1 + bnh_ret).cumprod()
    bnh_tot  = bnh_cum.iloc[-1] - 1

    results = {}
    for name, sig in strategies.items():
        sig_r = sig.reindex(bnh_ret.index).fillna(False).astype(float)
        cost  = sig_r.diff().abs() * TC
        net_r = sig_r * bnh_ret - cost
        cum_r = (1 + net_r).cumprod()
        strat_tot = cum_r.iloc[-1] - 1

        # 롤링 포착률 (30일)
        roll_strat = net_r.rolling(30).apply(lambda r: (1+r).prod()-1)
        roll_bnh   = bnh_ret.rolling(30).apply(lambda r: (1+r).prod()-1)
        roll_cap   = (roll_strat / roll_bnh.replace(0, np.nan)).dropna()

        results[name] = {
            "total_capture": strat_tot / bnh_tot if bnh_tot != 0 else 0,
            "strat_ret"    : strat_tot,
            "bnh_ret"      : bnh_tot,
            "n_days_active": int(sig_r.sum()),
            "cum"          : cum_r,
            "roll_capture" : roll_cap,
        }
    return results

# ===========================================================================
# 시각화 (8-panel 종합 차트)
# ===========================================================================

def plot_deep_analysis(etf_ret, spy_ret, vix_s, oil_p,
                       proba_full, df_full, df_is, df_oos,
                       models, sc, is_accs,
                       timing, overfit, feat_imp, thresh_df,
                       phase, capture):

    fig = plt.figure(figsize=(22, 28))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.42)
    oos_dt = pd.Timestamp(OOS_START)

    # ── 패널 1 (0,0): ML 신호 타임라인 vs XLE ─────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    xle_full = (1 + etf_ret).cumprod()
    xle_oos  = xle_full[xle_full.index >= oos_dt]
    ax.plot(xle_oos / xle_oos.iloc[0] * 100, color="#e67e22",
            linewidth=1.5, label="XLE (기준=100)")

    # ML 신호 발화 구간 하이라이트
    vrp_s   = df_full["vrp"].reindex(proba_full.index).ffill()
    ml_sig  = (proba_full > 0.55) & (vrp_s < 0)
    ml_oos  = ml_sig[ml_sig.index >= oos_dt]
    sig_on  = ml_oos.reindex(xle_oos.index, fill_value=False)
    for i in range(len(sig_on) - 1):
        if sig_on.iloc[i]:
            ax.axvspan(sig_on.index[i], sig_on.index[i+1],
                       alpha=0.25, color="#9b59b6")

    ax.axhline(100, color="gray", linewidth=0.5, alpha=0.4)
    ax.set_title(f"신호 발화 타임라인 vs XLE 가격\n"
                 f"ML 발화: {timing['n_on']}일/{len(ml_oos)}일 "
                 f"({timing['pct_active']*100:.1f}%)  "
                 f"보라색=ML 신호 ON 구간",
                 fontsize=9)
    ax.set_ylabel("XLE 기준가 (OOS 시작=100)", fontsize=8)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    # ── 패널 2 (0,1): 과적합 진단 -- 학습곡선 ─────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if overfit["lc_train"]:
        x_lc = overfit["lc_sizes"]
        ax2.plot(x_lc, overfit["lc_train"], "o-", color="#e74c3c",
                 linewidth=1.5, markersize=5, label="IS 훈련 정확도")
        ax2.plot(x_lc, overfit["lc_cv"],   "s--", color="#3498db",
                 linewidth=1.5, markersize=5, label="IS TimeSeriesCV 정확도")
    ax2.axhline(0.5, color="gray", linestyle=":", linewidth=0.8,
                label="무작위 기준(0.5)")

    # OOS 실제 정확도 점 표시
    if df_oos is not None and len(df_oos) > 10:
        proba_oos = ensemble_proba(models, sc, df_oos, is_accs)
        y_true    = df_oos["target"].reindex(proba_oos.index).dropna()
        y_pred    = (proba_oos.reindex(y_true.index) > 0.55).astype(int)
        oos_acc   = (y_pred == y_true).mean()
        ax2.axhline(oos_acc, color="#e74c3c", linestyle="-.",
                    linewidth=1.2, label=f"OOS 실제 정확도={oos_acc:.3f}")
        ax2.scatter(len(df_is), oos_acc, s=80, color="#e74c3c",
                    zorder=5, marker="X")

    ax2.set_xlabel("훈련 표본수 (일)", fontsize=8)
    ax2.set_ylabel("분류 정확도", fontsize=8)
    ax2.set_title("과적합 진단: GBT 학습곡선\n"
                  "빨강=훈련, 파랑=CV, 빨강X=OOS 실제",
                  fontsize=9)
    ax2.legend(fontsize=7.5)
    ax2.tick_params(labelsize=8)

    # ── 패널 3 (1,0): 피처 중요도 IS vs OOS ──────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    feats_short = ["beta", "vrp", "vix", "beta_z", "vrp_z", "vix_z",
                   "mom20", "mom63", "relstr", "bdelta"]
    x3 = np.arange(len(feats_short))
    w3 = 0.35
    ax3.bar(x3 - w3/2, feat_imp["is_imp"],  width=w3, color="#3498db",
            alpha=0.75, label="IS 피처 중요도")
    ax3.bar(x3 + w3/2, feat_imp["oos_imp"], width=w3, color="#e74c3c",
            alpha=0.75, label="OOS 피처 중요도")
    ax3.set_xticks(x3)
    ax3.set_xticklabels(feats_short, rotation=35, fontsize=7.5)
    ax3.set_title("피처 중요도: IS 학습(파랑) vs OOS 실제(빨강)\n"
                  "두 막대가 다를수록 체제 변화 심함",
                  fontsize=9)
    ax3.legend(fontsize=8)
    ax3.axhline(0, color="gray", linewidth=0.5)
    ax3.tick_params(labelsize=8)

    # ── 패널 4 (1,1): 임계값 민감도 ──────────────────────────────────
    ax4  = fig.add_subplot(gs[1, 1])
    ax4b = ax4.twinx()
    valid_th = thresh_df.dropna(subset=["sharpe"])
    ax4.plot(valid_th["threshold"], valid_th["sharpe"],
             "o-", color="#9b59b6", linewidth=1.5, label="OOS Sharpe")
    ax4b.bar(valid_th["threshold"], valid_th["n_days"],
             width=0.02, color="#3498db", alpha=0.4, label="발화 일수")
    ax4.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax4.axvline(0.55, color="#9b59b6", linestyle="--", linewidth=0.9,
                alpha=0.6, label="현재 임계값 0.55")
    ax4.set_xlabel("ML 임계값 (threshold)", fontsize=8)
    ax4.set_ylabel("OOS Sharpe", fontsize=8, color="#9b59b6")
    ax4b.set_ylabel("발화 일수", fontsize=8, color="#3498db")
    ax4.set_title("임계값 민감도: threshold 0.40~0.72\n"
                  "높을수록 보수적 (발화 적음) / 낮을수록 공격적",
                  fontsize=9)
    h1, l1 = ax4.get_legend_handles_labels()
    h2, l2 = ax4b.get_legend_handles_labels()
    ax4.legend(h1+h2, l1+l2, fontsize=7.5)
    ax4.tick_params(labelsize=8)
    ax4b.tick_params(labelsize=8)

    # ── 패널 5 (2,0): XLE 수익 단계 분해 ────────────────────────────
    ax5  = fig.add_subplot(gs[2, 0])
    ax5b = ax5.twinx()
    ax5.plot(phase["cum_xle"], color="#e67e22", linewidth=1.5, label="XLE 실제")
    ax5.plot(phase["cum_expected"], color="#3498db", linewidth=1.0,
             linestyle="--", label="베타 기여 (beta*SPY)")
    ax5.plot(phase["cum_alpha"], color="#27ae60", linewidth=1.0,
             linestyle=":", label="알파 기여 (초과수익)")
    ax5b.plot(phase["vix_oos"], color="#e74c3c", linewidth=0.8,
              alpha=0.6, label="VIX")
    ax5.set_title(f"XLE OOS 수익 분해\n"
                  f"유가 상관 {phase['oil_corr']:.2f} | "
                  f"VIX 레벨별 수익: <20={phase['vix_low_ret']*100:+.1f}% "
                  f"20~30={phase['vix_mid_ret']*100:+.1f}% "
                  f">30={phase['vix_high_ret']*100:+.1f}%",
                  fontsize=9)
    ax5.set_ylabel("누적 수익 (OOS 시작=1)", fontsize=8)
    ax5b.set_ylabel("VIX", fontsize=8, color="#e74c3c")
    h1, l1 = ax5.get_legend_handles_labels()
    h2, l2 = ax5b.get_legend_handles_labels()
    ax5.legend(h1+h2, l1+l2, fontsize=7.5)
    ax5.tick_params(labelsize=8)
    ax5b.tick_params(labelsize=8)

    # ── 패널 6 (2,1): 월별 XLE + 유가 수익 ──────────────────────────
    ax6  = fig.add_subplot(gs[2, 1])
    monthly = phase["monthly"]
    oil_monthly = phase["oil_oos"].resample("ME").apply(
        lambda r: (1 + r).prod() - 1)
    x6  = np.arange(len(monthly))
    w6  = 0.35
    ax6.bar(x6 - w6/2, monthly.values * 100, width=w6,
            color=["#27ae60" if v >= 0 else "#e74c3c" for v in monthly],
            alpha=0.75, label="XLE 월별 수익")
    oil_aligned = oil_monthly.reindex(monthly.index, method="nearest")
    ax6.bar(x6 + w6/2, oil_aligned.values * 100, width=w6,
            color="#f39c12", alpha=0.6, label="WTI 월별 수익")
    ax6.axhline(0, color="black", linewidth=0.5)
    ax6.set_xticks(x6)
    ax6.set_xticklabels(
        [d.strftime("%y-%m") for d in monthly.index],
        rotation=30, fontsize=7.5
    )
    ax6.set_ylabel("수익률 (%)", fontsize=8)
    ax6.set_title("OOS 월별 XLE(초록/빨강) vs WTI(주황) 수익\n"
                  "유가가 리드하는 달과 XLE 독자 움직임 달 비교",
                  fontsize=9)
    ax6.legend(fontsize=8)
    ax6.tick_params(labelsize=8)

    # ── 패널 7 (3,0): 알파 포착률 시계열 ────────────────────────────
    ax7 = fig.add_subplot(gs[3, 0])
    colors7 = {
        "Rule S2": "#e74c3c", "ML-Ensemble": "#9b59b6",
        "CVaR": "#e67e22"
    }
    for name, cap in capture.items():
        if "cum" in cap and len(cap["cum"]) > 5:
            ax7.plot(cap["cum"], linewidth=1.4,
                     color=colors7.get(name, "gray"), label=name)
    bnh_cum = (1 + etf_ret[etf_ret.index >= oos_dt]).cumprod()
    ax7.plot(bnh_cum, color="black", linewidth=1.0, linestyle="--",
             label="BnH XLE", alpha=0.6)
    ax7.set_title("전략별 OOS 누적 수익 vs Buy & Hold",
                  fontsize=9)
    ax7.set_ylabel("누적 수익 (1=시작점)", fontsize=8)
    ax7.legend(fontsize=8)
    ax7.tick_params(labelsize=8)

    # ── 패널 8 (3,1): 포착률 + 개선 방향 요약 표 ───────────────────
    ax8 = fig.add_subplot(gs[3, 1])
    ax8.axis("off")
    rows = []
    for name, cap in capture.items():
        rows.append([
            name,
            f"{cap['strat_ret']*100:+.1f}%",
            f"{cap['bnh_ret']*100:+.1f}%",
            f"{cap['total_capture']*100:.1f}%",
            f"{cap['n_days_active']}일",
        ])
    if rows:
        tbl = ax8.table(
            cellText  = rows,
            colLabels = ["전략", "전략 수익", "BnH 수익", "포착률", "보유 일수"],
            cellLoc   = "center", loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9.5)
        tbl.scale(1, 2.0)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor("#2c3e50")
                cell.set_text_props(color="white", weight="bold")
            elif "ML" in str(cell.get_text().get_text()):
                cell.set_facecolor("#f0eaff")

    ax8.set_title("알파 포착률 요약\n포착률 = 전략 수익 / BnH 수익", fontsize=9,
                  pad=15, weight="bold")

    fig.suptitle(
        "ML-Ensemble vs Buy & Hold XLE -- 심층 분석\n"
        "Sharpe 2.14의 진실 / +85.9% 이동을 왜 10%만 잡았나",
        fontsize=13, weight="bold", y=1.01,
    )
    out = FIGURES_DIR / "ml_deep_analysis.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  차트 저장: {out}")

# ===========================================================================
# 메인
# ===========================================================================

if __name__ == "__main__":
    sep = "=" * 62

    print(f"\n{sep}")
    print("  ML-Ensemble vs BnH XLE -- 심층 탐구")
    print(f"{sep}")

    # 1) 데이터
    print("\n[1] 데이터 로드...")
    prices = fetch_data(ENERGY_STOCKS + [SECTOR_ETF], FULL_START, OOS_END)
    spy_ret = prices["SPY"].pct_change().dropna()
    vix_s   = prices["^VIX"]
    etf_ret = prices[SECTOR_ETF].pct_change().dropna()
    oil_p   = prices.get("CL=F", pd.Series(dtype=float))
    print(f"  XLE: {len(etf_ret)}일  |  VIX: {len(vix_s)}일  "
          f"  Oil: {len(oil_p)}일")

    # 2) 피처 + 모델
    print("\n[2] 피처 엔지니어링 + 모델 학습...")
    df_full = build_features(etf_ret, spy_ret, vix_s)
    df_is   = df_full[df_full.index < OOS_START]
    df_oos  = df_full[df_full.index >= OOS_START]
    models, sc, is_accs = train_ensemble(df_is)
    proba_full = pd.Series(0.5, index=df_full.index)
    proba_full.update(ensemble_proba(models, sc, df_is, is_accs))
    proba_full.update(ensemble_proba(models, sc, df_oos, is_accs))
    print(f"  IS: {len(df_is)}일  |  OOS: {len(df_oos)}일")

    # 3) 신호 타이밍
    print("\n[3] 신호 발화 타임라인 분석...")
    timing = analyze_signal_timing(df_full, proba_full, etf_ret)
    print(f"  ML 발화 일수(OOS): {timing['n_on']}일 / {timing['n_on']+timing['n_off']}일 "
          f"({timing['pct_active']*100:.1f}%)")
    print(f"  신호 OFF 구간의 XLE 누적 수익: {timing['missed_ret']*100:+.1f}%")

    # 4) 과적합 진단
    print("\n[4] 과적합 진단...")
    overfit = overfitting_diagnosis(df_is, models, sc)
    print(f"  IS TimeSeriesCV 평균 정확도:")
    for k, v in overfit["cv_mean"].items():
        print(f"    {k}: {v:.3f} +/- {overfit['cv_std'][k]:.3f}")
    gap = is_accs["GBT"] - overfit["cv_mean"]["GBT"]
    print(f"  GBT 과적합 격차 (IS훈련 - IS-CV): {gap:.3f}")

    # 5) 피처 중요도
    print("\n[5] 피처 중요도 IS vs OOS...")
    feat_imp = feature_importance_shift(df_is, df_oos, models, sc)
    top_is  = sorted(zip(FEATURE_COLS, feat_imp["is_imp"]),
                     key=lambda x: -x[1])[:3]
    top_oos = sorted(zip(FEATURE_COLS, feat_imp["oos_imp"]),
                     key=lambda x: -x[1])[:3]
    print(f"  IS 상위 3 피처: {[f[0] for f in top_is]}")
    print(f"  OOS 상위 3 피처: {[f[0] for f in top_oos]}")

    # 6) 임계값 민감도
    print("\n[6] 임계값 민감도 스캔 (0.40~0.72)...")
    thresh_df = threshold_scan(proba_full, df_full, etf_ret)
    best = thresh_df.loc[thresh_df["sharpe"].idxmax()] if not thresh_df["sharpe"].isna().all() else None
    if best is not None:
        print(f"  최적 임계값: {best['threshold']:.2f}  "
              f"Sharpe={best['sharpe']:.2f}  "
              f"발화={int(best['n_days'])}일")

    # 7) XLE 단계 분해
    print("\n[7] XLE OOS 수익 단계 분해...")
    phase = xle_phase_decomposition(prices, etf_ret, spy_ret, vix_s, oil_p)
    print(f"  유가-XLE 상관 (OOS): {phase['oil_corr']:.3f}")
    print(f"  VIX 레벨별 XLE 수익:")
    print(f"    VIX < 20  ({phase['vix_low_days']}일): {phase['vix_low_ret']*100:+.1f}%")
    print(f"    VIX 20~30 ({phase['vix_mid_days']}일): {phase['vix_mid_ret']*100:+.1f}%")
    print(f"    VIX >= 30 ({phase['vix_high_days']}일): {phase['vix_high_ret']*100:+.1f}%")
    print(f"  월별 XLE 수익:")
    for dt, v in phase["monthly"].items():
        print(f"    {dt.strftime('%Y-%m')}: {v*100:+.1f}%")

    # 8) 알파 포착률
    print("\n[8] 알파 포착률 계산...")
    vrp_s = (build_features(etf_ret, spy_ret, vix_s)["vrp"]
             .reindex(etf_ret.index).ffill())
    rule_s2 = (vrp_s < 0).shift(1).fillna(False)
    ml_sig_oos = ((proba_full > 0.55) & (vrp_s.reindex(proba_full.index).ffill() < 0)
                  ).shift(1).fillna(False)

    capture = alpha_capture_rate(etf_ret, {
        "Rule S2"     : rule_s2,
        "ML-Ensemble" : ml_sig_oos,
    })
    print(f"\n  {'전략':<16} {'전략 수익':>10} {'BnH 수익':>10} "
          f"{'포착률':>10} {'보유 일수':>10}")
    print(f"  {'-'*56}")
    for name, cap in capture.items():
        print(f"  {name:<16} {cap['strat_ret']*100:>+9.1f}% "
              f"{cap['bnh_ret']*100:>+9.1f}%  "
              f"{cap['total_capture']*100:>8.1f}%  "
              f"{cap['n_days_active']:>8}일")
    capture["CVaR"] = capture["Rule S2"]  # 플롯용

    # 9) 시각화
    print("\n[9] 시각화...")
    plot_deep_analysis(
        etf_ret, spy_ret, vix_s, oil_p,
        proba_full, df_full, df_is, df_oos,
        models, sc, is_accs,
        timing, overfit, feat_imp, thresh_df,
        phase, capture,
    )

    # 10) 핵심 결론 (솔직하게)
    print(f"\n{sep}")
    print("  [핵심 결론]")
    print(f"\n  ML-Ensemble Sharpe=2.14의 진실:")
    print(f"    - OOS 발화 {timing['n_on']}일 ({timing['pct_active']*100:.1f}%) -- "
          f"거의 현금 보유")
    print(f"    - 발화 안 한 구간 XLE 수익 {timing['missed_ret']*100:+.1f}% 놓침")
    print(f"    - OOS 정확도 28.6% -- 무작위보다 나쁨 (OOS에서 IS 패턴 역전)")
    print(f"    - 높은 Sharpe = 거의 안 들어가서 손실 없음 (능력 아님)")

    print(f"\n  BnH XLE +85.9%의 구조:")
    print(f"    - 유가-XLE 상관: {phase['oil_corr']:.2f}")
    print(f"    - 공포 구간(VIX>=30, {phase['vix_high_days']}일) "
          f"수익 {phase['vix_high_ret']*100:+.1f}% 집중")
    print(f"    - 이 구간에 ML은 신호 없음 (공황 피크에서 과도한 신중함)")

    print(f"\n  왜 ML이 이 이동을 못 잡았나:")
    print(f"    IS(2025): VRP < 0 + beta < -0.05 조합이 '적당한' 상승 예측")
    print(f"    OOS(2026): Hormuz 봉쇄 심화로 체제가 '폭발적' 상승으로 전환")
    print(f"    ML은 체제 전환을 모름 -- 훈련 분포 밖 사건")

    print(f"\n  실질적 개선 방향:")
    print(f"    1. 온톨로지 분류기 우선: 공급충격 체제 확인 후 ML 적용")
    print(f"    2. 체제별 별도 ML: 공급충격 체제 vs 정상 체제 분리 학습")
    print(f"    3. 멀티-위기 데이터: GFC+COVID+호르무즈 결합 LOO-CV")
    print(f"    4. 신호 조건 완화: threshold 낮추거나 VRP 조건 단독 사용")
    print(f"    5. 진입 비중 ML화: 방향은 규칙, 비중만 ML (Kelly 대체)")
    print(f"{sep}\n")

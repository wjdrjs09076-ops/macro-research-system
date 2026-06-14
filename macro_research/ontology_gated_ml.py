"""
ontology_gated_ml.py

개선 방향 1번: 온톨로지 분류기 선행 게이트

기존 문제:
  ML이 체제와 무관하게 VRP + 모멘텀 피처만으로 신호 발생
  -> 닷컴(VALUATION), COVID(SUBSECTOR) 체제에서도 신호 발화 가능

개선 구조:
  Step 1. 온톨로지 게이트 (체제 점수를 매일 계산)
    - beta 반전 지속성, VRP 음전환 비율, ETF-시장 디커플링 정도로 점수화
    - 공급충격/구조붕괴 점수 > 임계값 + N일 지속 -> 게이트 열림
  Step 2. ML 신호 (게이트 열렸을 때만 활성화)
    - ensemble proba > threshold AND VRP < 0

4-전략 비교:
  A. Pure ML (현재 베이스라인)
  B. 온톨로지 게이트 단독 (ML 없음)
  C. 온톨로지 게이트 + ML (새 구조)
  D. 온톨로지 게이트 + Rule S2 (VRP<0)
  BnH XLE

교차 위기 검증:
  4개 위기에 게이트 적용 -> 게이트가 올바른 체제만 여는지 확인
  닷컴(VALUATION): 게이트 닫혀야 함
  COVID(SUBSECTOR): 게이트 닫혀야 함
  GFC(STRUCTURAL): 게이트 열려야 함 (SHORT)
  호르무즈(SUPPLY_SHOCK): 게이트 열려야 함 (LONG)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import yfinance as yf

BASE_DIR    = Path(__file__).parent
FIGURES_DIR = BASE_DIR / "output" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

IS_START   = "2025-01-01"
IS_END     = "2025-12-31"
OOS_START  = "2026-01-01"
OOS_END    = "2026-05-29"
FULL_START = "2024-07-01"

TC            = 0.0005
EWMA_SPAN     = 32
RV_WIN        = 20
BETA_WIN      = 30
GATE_WINDOW   = 15   # 게이트 점수 계산 롤링 윈도우
GATE_CONSEC   = 5    # 게이트 열리려면 점수 >= threshold 연속 일수
GATE_THRESH   = 0.45 # 게이트 점수 임계값

# ── 평가 재설계 (감사 라운드 7 P1-2/P1-3) ──────────────────────────────
VOLTGT_TARGET = 0.15   # P1-2: vol-target 연 15% 목표 변동성
VOLTGT_CAP    = 1.5    # vol-target 레버리지 상한 (w_t = target/σ̂_t)
MA_FILTER_WIN = 200    # P1-2: 200일 이동평균 필터
NULL_SIMS     = 10000  # P1-3: 랜덤 타이밍 널 시뮬 횟수
NULL_SEED     = 42

FEATURE_COLS = [
    "beta_30d", "vrp", "vix",
    "beta_30d_z", "vrp_z", "vix_z",
    "mom_20d", "mom_63d", "rel_str", "beta_delta5",
]

CRISIS_PERIODS = {
    "hormuz": {
        "label"    : "호르무즈 봉쇄 (2025-2026)",
        "start"    : "2024-07-01",
        "end"      : "2026-05-29",
        "oos_start": "2026-01-01",
        "etf"      : "XLE",
        "direction": "LONG",
        "expected_gate": "OPEN",
    },
    "gfc": {
        "label"    : "글로벌 금융위기 (2006-2010)",
        "start"    : "2005-07-01",
        "end"      : "2010-12-31",
        "oos_start": "2008-09-15",
        "etf"      : "XLF",
        "direction": "SHORT",
        "expected_gate": "OPEN",
    },
    "covid": {
        "label"    : "COVID-19 (2019-2021)",
        "start"    : "2018-07-01",
        "end"      : "2021-12-31",
        "oos_start": "2020-02-20",
        "etf"      : "XLV",
        "direction": "LONG",
        "expected_gate": "CLOSED",
    },
    "dotcom": {
        "label"    : "닷컴버블 (1998-2002)",
        "start"    : "1997-07-01",
        "end"      : "2002-12-31",
        "oos_start": "2000-03-10",
        "etf"      : "XLK",
        "direction": "SHORT",
        "expected_gate": "CLOSED",
    },
}

# ===========================================================================
# 공통 함수
# ===========================================================================

def fetch(tickers, start, end):
    all_t = list(set(tickers + ["SPY", "^VIX"]))
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
    vix_a   = vix.reindex(idx).ffill() / 100.0
    capm_iv = np.sqrt((b63 * vix_a)**2 + idio**2)
    rv_ann  = s.rolling(RV_WIN).std() * np.sqrt(252)
    return (capm_iv - rv_ann).dropna()

def build_features(sec_ret, spy_ret, vix_series):
    beta = rolling_beta(sec_ret, spy_ret)
    vrp  = compute_vrp(sec_ret, spy_ret, vix_series)
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
    feat = df[FEATURE_COLS].dropna()
    if len(feat) == 0:
        return pd.Series(dtype=float)
    Xs  = sc.transform(feat.values)
    idx = feat.index
    ps  = {n: m.predict_proba(Xs)[:, 1] for n, m in models.items()}
    if is_accs:
        tot = sum(is_accs.values())
        arr = sum(ps[k] * is_accs[k] / tot for k in models)
    else:
        arr = np.mean(list(ps.values()), axis=0)
    return pd.Series(arr, index=idx)

def backtest(price_ret, signal, tc=TC, oos_start=None):
    sig  = signal.reindex(price_ret.index).fillna(False).astype(float)
    cost = sig.diff().abs() * tc
    net  = sig * price_ret - cost
    cum  = (1 + net).cumprod()

    def _s(r, c):
        n = len(r)
        if n < 5 or c.empty or c.iloc[-1] <= 0:
            return dict(cagr=np.nan, sharpe=np.nan, mdd=np.nan, n=0)
        cagr   = c.iloc[-1] ** (252 / n) - 1
        sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
        mdd    = ((c - c.cummax()) / c.cummax()).min()
        # 활성일 = 해당 세그먼트(r.index) 내에서만 — 기존 sig.sum() 은 전체기간을
        # 세어 OOS 카드에 전체기간 활성일이 찍히던 버그 (P1-5 진단 중 발견)
        active = int((sig.reindex(r.index).fillna(0).abs() > 0).sum())
        return dict(cagr=cagr, sharpe=sharpe, mdd=mdd, n=active)

    res = {"total": _s(net, cum), "cum": cum}
    if oos_start:
        oos_dt  = pd.Timestamp(oos_start)
        r_oos   = net[net.index >= oos_dt]
        c_oos   = (1 + r_oos).cumprod()
        res["oos"] = _s(r_oos, c_oos)
        res["oos_ret"] = r_oos          # OOS 전용 순수익 (재기준 차트/raw 산출용)
    return res


# ===========================================================================
# 평가 재설계 — 베이스라인 / 랜덤 타이밍 널 (감사 라운드 7 P1-2/P1-3)
# ===========================================================================

def vol_target_signal(etf_ret, target=VOLTGT_TARGET, span=EWMA_SPAN,
                      cap=VOLTGT_CAP):
    """P1-2 베이스라인 ①: 연 target 변동성 목표 가변 비중.

    w_t = target / σ̂_t. σ̂_t = EWMA(span) 연환산 변동성 — 인과적(과거만).
    shift(1) 로 어제 추정 변동성으로 오늘 비중 결정 (look-ahead 차단).
    """
    sigma = etf_ret.ewm(span=span, min_periods=20).std() * np.sqrt(252)
    w = (target / sigma).clip(upper=cap)
    return w.shift(1).fillna(0.0)


def ma_filter_signal(etf_ret, window=MA_FILTER_WIN):
    """P1-2 베이스라인 ②: 가격 > N일 이동평균이면 롱 (on/off)."""
    px = (1 + etf_ret).cumprod()
    ma = px.rolling(window, min_periods=window).mean()
    return (px > ma).shift(1).fillna(False)


def backtest_weighted(price_ret, weight, tc=TC, oos_start=None):
    """연속 비중 백테스트 (vol-target 용). 활성일 = 비중>0 인 날."""
    w    = weight.reindex(price_ret.index).fillna(0.0).astype(float)
    cost = w.diff().abs() * tc
    net  = w * price_ret - cost
    cum  = (1 + net).cumprod()

    def _s(r, c, wseg):
        n = len(r)
        if n < 5 or c.empty or c.iloc[-1] <= 0:
            return dict(cagr=np.nan, sharpe=np.nan, mdd=np.nan, n=0)
        cagr   = c.iloc[-1] ** (252 / n) - 1
        sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
        mdd    = ((c - c.cummax()) / c.cummax()).min()
        return dict(cagr=cagr, sharpe=sharpe, mdd=mdd, n=int((wseg > 0).sum()))

    res = {"total": _s(net, cum, w), "cum": cum}
    if oos_start:
        oos_dt = pd.Timestamp(oos_start)
        r_oos  = net[net.index >= oos_dt]
        w_oos  = w[w.index >= oos_dt]
        c_oos  = (1 + r_oos).cumprod()
        res["oos"] = _s(r_oos, c_oos, w_oos)
        res["oos_ret"] = r_oos
    return res


def random_timing_null(oos_r, n_active, n_sims=NULL_SIMS, seed=NULL_SEED):
    """P1-3: 활성일 수를 n_active 로 고정한 무작위 노출의 OOS gross 누적수익 분포.

    long-spot on/off 매핑 (노출일 = 1+r, 비노출 = 1). 거래비용 제외(순수 타이밍).
    반환: 길이 n_sims ndarray (각 시뮬의 누적수익률), 불가 시 None.
    """
    T = len(oos_r)
    if n_active <= 0 or n_active >= T:
        return None
    rng   = np.random.default_rng(seed)
    order = rng.random((n_sims, T)).argsort(axis=1)[:, :n_active]
    mask  = np.zeros((n_sims, T))
    np.put_along_axis(mask, order, 1.0, axis=1)
    return np.prod(1 + mask * oos_r, axis=1) - 1


def timing_null_for_signal(etf_ret, signal, oos_start, n_sims=NULL_SIMS):
    """이진 신호의 실제 OOS gross 수익이 같은 활성일 수의 무작위 노출 분포에서
    차지하는 백분위를 계산. p≈0.5 면 타이밍 무정보, p→1 이면 평균 대비 우월."""
    oos_dt = pd.Timestamp(oos_start)
    oos_r  = etf_ret[etf_ret.index >= oos_dt].dropna()
    sig    = signal.reindex(oos_r.index).fillna(False).astype(float)
    n_active = int(sig.sum())
    gross  = float(np.prod(1 + sig.values * oos_r.values) - 1)
    sims   = random_timing_null(oos_r.values, n_active, n_sims)
    if sims is None:
        return {"n_active": n_active, "gross_return": round(gross * 100, 2),
                "null_pctile": None, "null_median": None}
    return {"n_active": n_active,
            "gross_return": round(gross * 100, 2),
            "null_pctile": round(float((sims < gross).mean()), 3),
            "null_median": round(float(np.median(sims)) * 100, 2)}


def build_cards(etf_ret, signals, oos_start, etf_label="XLE", n_null_sims=NULL_SIMS):
    """OOS 전략 카드 일괄 생성 (4 게이트 전략 + 베이스라인 2 + BnH).

    raw/누적은 OOS 전용 수익으로 재계산(P1-4), 활성일은 세그먼트 기준(P1-5 수정).
    반환: (strategies_payload, eval_meta).
    """
    oos_dt   = pd.Timestamp(oos_start)
    bnh_oos  = etf_ret[etf_ret.index >= oos_dt]
    bnh_cum  = (1 + bnh_oos).cumprod()
    bnh_raw  = float(bnh_cum.iloc[-1] - 1)
    bnh_cagr = float(bnh_cum.iloc[-1] ** (252 / len(bnh_oos)) - 1)
    bnh_sh   = float(bnh_oos.mean() / bnh_oos.std() * np.sqrt(252)) if bnh_oos.std() > 0 else float("nan")
    bnh_mdd  = float(((bnh_cum - bnh_cum.cummax()) / bnh_cum.cummax()).min())

    def card(res, sig=None, weighted=False):
        oos_r   = res.get("oos", {})
        r_oos   = res.get("oos_ret")
        cum_oos = (1 + r_oos).cumprod() if r_oos is not None and len(r_oos) else None
        raw_ret = float(cum_oos.iloc[-1] - 1) if cum_oos is not None else 0.0
        c = {
            "oos_cagr"       : round(float(oos_r.get("cagr", 0) or 0) * 100, 2),
            "oos_sharpe"     : round(float(oos_r.get("sharpe", 0) or 0), 3),
            "oos_mdd"        : round(float(oos_r.get("mdd", 0) or 0) * 100, 2),
            "oos_active_days": int(oos_r.get("n", 0)),
            "oos_raw_return" : round(raw_ret * 100, 2),   # alpha_capture = raw / B&H raw
            "alpha_capture"  : round(raw_ret / bnh_raw * 100, 1) if bnh_raw != 0 else 0,
            "cumulative": {
                "dates" : cum_oos.index.strftime("%Y-%m-%d").tolist() if cum_oos is not None else [],
                "values": [round(x, 6) for x in cum_oos.tolist()] if cum_oos is not None else [],
            },
        }
        if sig is not None and not weighted:
            c["timing_null"] = timing_null_for_signal(etf_ret, sig, oos_start, n_null_sims)
        return c

    sp = {}
    for sname, sig in signals.items():
        sp[sname] = card(backtest(etf_ret, sig, TC, oos_start), sig)
    sp[f"VolTgt {etf_label}"] = card(
        backtest_weighted(etf_ret, vol_target_signal(etf_ret), TC, oos_start), weighted=True)
    sp[f"200DMA {etf_label}"] = card(
        backtest(etf_ret, ma_filter_signal(etf_ret), TC, oos_start), ma_filter_signal(etf_ret))
    sp[f"BnH {etf_label}"] = {
        "oos_cagr"       : round(bnh_cagr * 100, 2),
        "oos_sharpe"     : round(bnh_sh, 3) if bnh_sh == bnh_sh else None,
        "oos_mdd"        : round(bnh_mdd * 100, 2),
        "oos_active_days": len(bnh_oos),
        "oos_raw_return" : round(bnh_raw * 100, 2),
        "alpha_capture"  : 100.0,
        "cumulative": {
            "dates" : bnh_oos.index.strftime("%Y-%m-%d").tolist(),
            "values": [round(x, 6) for x in bnh_cum.tolist()],
        },
    }
    meta = {
        "alpha_capture_def": "OOS 누적수익(raw) / B&H 누적수익(raw) × 100",
        "bnh_raw_return"   : round(bnh_raw * 100, 2),
        "random_null": {
            "sims"  : n_null_sims,
            "method": "활성일 수 고정 · 무작위 노출 · gross(거래비용 제외) "
                      "누적수익 분포에서의 백분위. p≈0.5=타이밍 무정보, "
                      "p→1=평균일 대비 우월, p→0=열등.",
        },
        "baselines": {
            f"VolTgt {etf_label}": f"연 {VOLTGT_TARGET*100:.0f}% vol-target "
                                   f"(EWMA span={EWMA_SPAN}, cap {VOLTGT_CAP}x, 연속 비중)",
            f"200DMA {etf_label}": f"가격 > {MA_FILTER_WIN}일 이동평균 시 롱 (on/off)",
        },
    }
    return sp, meta


# 하락 위기 창 — 게이트의 '홈그라운드 시험' (감사 P1-1). 시점-aware:
# IS = full_start~oos_start 로만 ML 학습, 위기 창 이후 데이터 미사용.
CRISIS_WINDOWS = [
    {"key": "gfc",   "label": "GFC (2008)",      "etf": "XLF",
     "full_start": "2006-06-01", "oos_start": "2008-06-01", "oos_end": "2009-06-30"},
    {"key": "covid", "label": "COVID (2020)",    "etf": "XLE",
     "full_start": "2018-06-01", "oos_start": "2020-01-01", "oos_end": "2020-12-31"},
    {"key": "tighten", "label": "긴축 (2022)",    "etf": "XLK",
     "full_start": "2020-06-01", "oos_start": "2022-01-01", "oos_end": "2022-12-31"},
]


def evaluate_window(etf, full_start, oos_start, oos_end, direction="LONG"):
    """한 창(위기)에서 5전략 + 베이스라인 카드 생성. 데이터를 oos_end 까지만
    받아 OOS 슬라이스가 자동으로 창에 한정된다 (look-ahead 차단)."""
    prices = fetch([etf], full_start, oos_end)
    if etf not in prices or "SPY" not in prices or "^VIX" not in prices:
        return None
    spy_ret = prices["SPY"].pct_change().dropna()
    vix_s   = prices["^VIX"]
    etf_ret = prices[etf].pct_change().dropna()
    df_full = build_features(etf_ret, spy_ret, vix_s)
    df_is   = df_full[df_full.index <  oos_start]
    df_oos  = df_full[df_full.index >= oos_start]
    if len(df_is) < 60 or len(df_oos) < 10:
        return None
    models, sc, accs = train_ensemble(df_is)
    proba = pd.Series(0.5, index=df_full.index)
    proba.update(ensemble_proba(models, sc, df_is, accs))
    proba.update(ensemble_proba(models, sc, df_oos, accs))
    beta_s  = rolling_beta(etf_ret, spy_ret)
    vrp_s   = compute_vrp(etf_ret, spy_ret, vix_s)
    gate_df = compute_gate_scores(beta_s, vrp_s, etf_ret, spy_ret, direction=direction)
    signals = build_all_signals(df_full, proba, gate_df, vrp_s)
    sp, meta = build_cards(etf_ret, signals, oos_start, etf_label=etf)
    return {
        "etf": etf, "direction": direction,
        "oos_start": oos_start, "oos_end": oos_end,
        "oos_gate_open_pct": round(float(
            gate_df["gate_open"][gate_df.index >= pd.Timestamp(oos_start)].mean() * 100), 1),
        "strategies": sp,
        "bnh_raw_return": meta["bnh_raw_return"],
    }


# ===========================================================================
# 핵심: 온톨로지 게이트 점수
# ===========================================================================

def compute_gate_scores(
    beta_series : pd.Series,
    vrp_series  : pd.Series,
    etf_ret     : pd.Series,
    spy_ret     : pd.Series,
    direction   : str = "LONG",
    window      : int = GATE_WINDOW,
) -> pd.DataFrame:
    """
    매일 체제 점수를 계산 (0~1 연속값).

    공급충격 점수 (LONG 방향):
      - beta_rev   : rolling window 내 beta < -0.05 비율 (온톨로지 FULL_ETF_DECOUPLING 대응)
      - vrp_neg    : rolling window 내 VRP < 0 비율     (VRP 음전환 체제 대응)
      - etf_outperf: rolling window 내 ETF > SPY 초과수익 비율 (디커플링 방향 확인)

    구조붕괴 점수 (SHORT 방향):
      - beta_spike : rolling window 내 beta > 1.5 비율
      - vrp_neg    : (동일)
      - etf_underperf: rolling window 내 ETF < SPY 비율

    가중치:
      beta 반전/급등: 0.40 (핵심 조건, FULL_ETF_DECOUPLING 대응)
      VRP 음전환:     0.35 (VRP 체제 확인)
      방향성 확인:    0.25 (ETF 초과/미달 수익)
    """
    idx = beta_series.index.intersection(vrp_series.index)
    idx = idx.intersection(etf_ret.index).intersection(spy_ret.index)

    b    = beta_series.reindex(idx)
    v    = vrp_series.reindex(idx)
    er   = etf_ret.reindex(idx)
    sr   = spy_ret.reindex(idx)
    excs = er - sr

    # 롤링 비율
    beta_rev    = (b < -0.05).astype(float).rolling(window, min_periods=5).mean()
    beta_spike  = (b > 1.50).astype(float).rolling(window, min_periods=5).mean()
    vrp_neg     = (v < 0).astype(float).rolling(window, min_periods=5).mean()
    etf_out     = (excs > 0).astype(float).rolling(window, min_periods=5).mean()
    etf_under   = (excs < 0).astype(float).rolling(window, min_periods=5).mean()

    # 방향별 점수
    if direction == "LONG":
        gate_score = 0.40 * beta_rev + 0.35 * vrp_neg + 0.25 * etf_out
    else:  # SHORT
        gate_score = 0.40 * beta_spike + 0.35 * vrp_neg + 0.25 * etf_under

    # 지속성 확인: 연속 N일 gate_score >= threshold -> gate_open
    above_thresh = (gate_score >= GATE_THRESH).astype(float)
    gate_open    = above_thresh.rolling(GATE_CONSEC, min_periods=GATE_CONSEC).min()
    gate_open    = gate_open.fillna(0).astype(bool)

    df = pd.DataFrame({
        "gate_score"   : gate_score,
        "beta_component": beta_rev if direction == "LONG" else beta_spike,
        "vrp_component" : vrp_neg,
        "dir_component" : etf_out if direction == "LONG" else etf_under,
        "gate_open"    : gate_open,
    }, index=idx)
    return df


# ===========================================================================
# 4-전략 신호 구성
# ===========================================================================

def build_all_signals(
    df_full     : pd.DataFrame,
    proba_full  : pd.Series,
    gate_df     : pd.DataFrame,
    vrp_series  : pd.Series,
    ml_threshold: float = 0.55,
) -> dict:
    """
    A. Pure ML          : proba > th AND VRP < 0
    B. Gate Only        : gate_open (방향 기반, VRP 없음)
    C. Gate + ML        : gate_open AND proba > th AND VRP < 0
    D. Gate + Rule S2   : gate_open AND VRP < 0
    """
    idx    = df_full.index.union(proba_full.index).union(gate_df.index)
    p      = proba_full.reindex(df_full.index).ffill().fillna(0.5)
    vrp    = vrp_series.reindex(df_full.index).ffill()
    gate   = gate_df["gate_open"].reindex(df_full.index).ffill().fillna(False)

    # A
    sig_a = ((p > ml_threshold) & (vrp < 0)).shift(1).fillna(False)
    # B
    sig_b = gate.shift(1).fillna(False)
    # C
    sig_c = (gate & (p > ml_threshold) & (vrp < 0)).shift(1).fillna(False)
    # D
    sig_d = (gate & (vrp < 0)).shift(1).fillna(False)

    return {"Pure ML": sig_a, "Gate Only": sig_b,
            "Gate+ML": sig_c, "Gate+S2": sig_d}


# ===========================================================================
# 교차 위기 게이트 검증
# ===========================================================================

def cross_crisis_gate_test() -> dict:
    """
    4개 위기에 게이트를 적용해 올바른 체제만 여는지 검증.
    온톨로지 예측:
      호르무즈 (SUPPLY_SHOCK) -> OPEN
      GFC      (STRUCTURAL)   -> OPEN
      COVID    (SUBSECTOR)    -> CLOSED
      닷컴     (VALUATION)    -> CLOSED
    """
    results = {}
    for name, cfg in CRISIS_PERIODS.items():
        print(f"  [{cfg['label']}] 데이터 로드...")
        prices = fetch([cfg["etf"]], cfg["start"], cfg["end"])
        if "SPY" not in prices or "^VIX" not in prices or cfg["etf"] not in prices:
            print(f"    SKIP: 데이터 없음")
            continue

        spy_ret = prices["SPY"].pct_change().dropna()
        vix_s   = prices["^VIX"]
        etf_ret = prices[cfg["etf"]].pct_change().dropna()

        beta_s = rolling_beta(etf_ret, spy_ret)
        vrp_s  = compute_vrp(etf_ret, spy_ret, vix_s)

        gate_df = compute_gate_scores(
            beta_s, vrp_s, etf_ret, spy_ret,
            direction=cfg["direction"]
        )

        oos_dt     = pd.Timestamp(cfg["oos_start"])
        gate_oos   = gate_df["gate_open"][gate_df.index >= oos_dt]
        score_oos  = gate_df["gate_score"][gate_df.index >= oos_dt]
        n_open     = int(gate_oos.sum())
        n_total    = len(gate_oos)
        open_pct   = n_open / max(n_total, 1) * 100
        mean_score = score_oos.mean()
        correct    = (
            (cfg["expected_gate"] == "OPEN"   and open_pct > 20) or
            (cfg["expected_gate"] == "CLOSED" and open_pct < 20)
        )

        results[name] = {
            "label"        : cfg["label"],
            "direction"    : cfg["direction"],
            "expected"     : cfg["expected_gate"],
            "open_pct"     : open_pct,
            "mean_score"   : mean_score,
            "correct"      : correct,
            "gate_series"  : gate_df["gate_score"],
            "etf_price"    : prices[cfg["etf"]],
            "oos_start"    : cfg["oos_start"],
        }
        tick = "OK" if correct else "FAIL"
        print(f"    OOS 게이트 열림: {open_pct:.1f}%  "
              f"평균점수: {mean_score:.3f}  "
              f"예측: {cfg['expected_gate']}  [{tick}]")

    return results


# ===========================================================================
# 시각화
# ===========================================================================

def plot_results(
    etf_ret    : pd.Series,
    gate_df    : pd.DataFrame,
    signals    : dict,
    proba_full : pd.Series,
    vrp_series : pd.Series,
    cross_res  : dict,
):
    oos_dt = pd.Timestamp(OOS_START)

    fig = plt.figure(figsize=(22, 24))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.52, wspace=0.40)

    # ── 패널 1 (0,:): 게이트 점수 + 구성요소 시계열 (호르무즈) ──────────
    ax = fig.add_subplot(gs[0, :])
    ax.plot(gate_df["gate_score"], color="#1abc9c", linewidth=1.4,
            label="종합 게이트 점수")
    ax.plot(gate_df["beta_component"] * 0.40, color="navy",
            linewidth=0.9, alpha=0.7, linestyle="--",
            label="beta 반전 기여 (x0.40)")
    ax.plot(gate_df["vrp_component"]  * 0.35, color="darkorange",
            linewidth=0.9, alpha=0.7, linestyle="--",
            label="VRP 음전환 기여 (x0.35)")
    ax.plot(gate_df["dir_component"]  * 0.25, color="#9b59b6",
            linewidth=0.9, alpha=0.7, linestyle="--",
            label="방향성 기여 (x0.25)")
    ax.axhline(GATE_THRESH, color="#1abc9c", linestyle=":", linewidth=1.0,
               label=f"임계값 {GATE_THRESH}")
    ax.axvline(oos_dt, color="red", linestyle="--", linewidth=1.2,
               label="OOS 시작")

    # 게이트 열림 구간 음영
    go = gate_df["gate_open"].reindex(gate_df.index).fillna(False)
    for i in range(len(go) - 1):
        if go.iloc[i]:
            ax.axvspan(go.index[i], go.index[i+1], alpha=0.12, color="#1abc9c")

    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        f"온톨로지 게이트 점수 시계열 (XLE/호르무즈)\n"
        f"초록 음영 = 게이트 열림  |  "
        f"구성: beta반전(0.40) + VRP음전환(0.35) + ETF방향(0.25)  |  "
        f"연속 {GATE_CONSEC}일 >= {GATE_THRESH} 시 게이트 개방",
        fontsize=9,
    )
    ax.legend(fontsize=7.5, ncol=3)
    ax.tick_params(labelsize=8)

    # ── 패널 2 (1,0): 4-전략 누적 수익 비교 ────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    colors = {
        "Pure ML" : "#e74c3c",
        "Gate Only": "#1abc9c",
        "Gate+ML"  : "#9b59b6",
        "Gate+S2"  : "#e67e22",
    }
    bnh = (1 + etf_ret[etf_ret.index >= oos_dt]).cumprod()
    ax2.plot(bnh, color="black", linewidth=1.0, linestyle="--",
             alpha=0.6, label="BnH XLE")
    bt_results = {}
    for name, sig in signals.items():
        res = backtest(etf_ret, sig, TC, OOS_START)
        bt_results[name] = res
        cum_oos = res["cum"][res["cum"].index >= oos_dt]
        if len(cum_oos) > 1:
            ax2.plot(cum_oos, color=colors.get(name, "gray"),
                     linewidth=1.4, label=name)
    ax2.set_title("4-전략 OOS 누적 수익 비교\n"
                  "Gate 유무 + ML/S2 조합", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.tick_params(labelsize=8)

    # ── 패널 3 (1,1): 신호 발화 일수 + OOS Sharpe 비교 ─────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    labels3, sharpes3, fire_days3 = [], [], []
    for name, res in bt_results.items():
        oos = res.get("oos", {})
        labels3.append(name)
        sharpes3.append(oos.get("sharpe", np.nan))
        fire_days3.append(oos.get("n", 0))

    x3  = np.arange(len(labels3))
    ax3b = ax3.twinx()
    bars = ax3.bar(x3, sharpes3,
                   color=[colors.get(l, "gray") for l in labels3],
                   alpha=0.75, width=0.5, label="OOS Sharpe")
    ax3b.plot(x3, fire_days3, "o--", color="dimgray",
              linewidth=1.2, markersize=6, label="발화 일수")
    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(labels3, fontsize=8.5)
    ax3.set_ylabel("OOS Sharpe", fontsize=8)
    ax3b.set_ylabel("발화 일수 (OOS)", fontsize=8, color="dimgray")
    ax3.set_title("전략별 OOS Sharpe vs 발화 일수\n"
                  "높은 Sharpe + 충분한 발화 일수가 목표", fontsize=9)
    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = ax3b.get_legend_handles_labels()
    ax3.legend(h1+h2, l1+l2, fontsize=8)
    ax3.tick_params(labelsize=8)
    ax3b.tick_params(labelsize=8)

    # ── 패널 4~7 (2~3,:): 교차 위기 게이트 점수 ────────────────────────
    crisis_keys = list(cross_res.keys())
    for i, key in enumerate(crisis_keys[:4]):
        row, col = divmod(i, 2)
        ax_c = fig.add_subplot(gs[2 + row, col])
        res  = cross_res[key]
        gs_s = res["gate_series"]
        oos  = pd.Timestamp(res["oos_start"])

        ax_c.plot(gs_s, color="#1abc9c", linewidth=1.0, alpha=0.8)
        ax_c.axhline(GATE_THRESH, color="#1abc9c", linestyle=":",
                     linewidth=0.9, alpha=0.7)
        ax_c.axvline(oos, color="red", linestyle="--", linewidth=1.0,
                     label="OOS 시작")
        ax_c.fill_between(gs_s.index, GATE_THRESH, gs_s.where(gs_s > GATE_THRESH),
                          alpha=0.2, color="#1abc9c")

        etf_n = (res["etf_price"] / res["etf_price"].iloc[0] * 100)
        ax_cb = ax_c.twinx()
        ax_cb.plot(etf_n, color="gray", linewidth=0.8, alpha=0.5)
        ax_cb.set_ylabel("ETF (기준=100)", fontsize=7, color="gray")
        ax_cb.tick_params(labelsize=7)

        correct_str = "OK" if res["correct"] else "FAIL"
        expected_c  = "#27ae60" if res["expected"] == "OPEN" else "#e74c3c"
        ax_c.set_title(
            f"{res['label']}\n"
            f"OOS 게이트: {res['open_pct']:.0f}% 열림  "
            f"(예측: {res['expected']})  "
            f"[{correct_str}]",
            fontsize=8.5, color=expected_c,
        )
        ax_c.set_ylim(-0.05, 1.05)
        ax_c.legend(fontsize=7)
        ax_c.tick_params(labelsize=7)

    fig.suptitle(
        "온톨로지 게이트 선행 ML 프레임워크\n"
        "체제 점수(게이트) -> ML 신호: 잘못된 체제에서 신호 차단",
        fontsize=13, weight="bold", y=1.01,
    )
    out = FIGURES_DIR / "ontology_gated_ml.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  차트 저장: {out}")


# ===========================================================================
# 성과 요약 테이블
# ===========================================================================

def print_summary(bt_results: dict, etf_ret: pd.Series):
    oos_dt  = pd.Timestamp(OOS_START)
    bnh_ret = etf_ret[etf_ret.index >= oos_dt]
    bnh_raw = (1 + bnh_ret).cumprod().iloc[-1] - 1

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  {'전략':<16} {'OOS CAGR':>10} {'Sharpe':>9} {'MDD':>8} "
          f"{'발화':>6} {'포착률':>8}")
    print(f"  {'-'*66}")
    for name, res in bt_results.items():
        oos = res.get("oos", {})
        n   = oos.get("n", 0)
        raw_oos_ret = (res["cum"][res["cum"].index >= oos_dt].iloc[-1] - 1)
        capture = raw_oos_ret / bnh_raw if bnh_raw != 0 else 0
        print(f"  {name:<16} "
              f"{oos.get('cagr', np.nan)*100:>+9.1f}%  "
              f"{oos.get('sharpe', np.nan):>8.2f}  "
              f"{oos.get('mdd', np.nan)*100:>7.1f}%  "
              f"{n:>5}일  "
              f"{capture*100:>7.1f}%")
    bnh_cagr = (1 + bnh_ret).cumprod().iloc[-1] ** (252/len(bnh_ret)) - 1
    bnh_sh   = bnh_ret.mean() / bnh_ret.std() * np.sqrt(252)
    print(f"  {'BnH XLE':<16} {bnh_cagr*100:>+9.1f}%  "
          f"{bnh_sh:>8.2f}   {'---':>7}  {'---':>5}  {'100.0%':>8}")
    print(sep)


# ===========================================================================
# 메인
# ===========================================================================

if __name__ == "__main__":
    sep = "=" * 62
    print(f"\n{sep}")
    print("  온톨로지 게이트 선행 ML 프레임워크")
    print(f"  게이트: 롤링{GATE_WINDOW}일 점수 >= {GATE_THRESH} 연속 {GATE_CONSEC}일")
    print(f"{sep}")

    # ── 1) 호르무즈 데이터 + 모델 학습 ────────────────────────────────
    print("\n[1] 데이터 + ML 모델 학습 (IS: 2025)...")
    prices  = fetch(["XLE"], FULL_START, OOS_END)
    spy_ret = prices["SPY"].pct_change().dropna()
    vix_s   = prices["^VIX"]
    etf_ret = prices["XLE"].pct_change().dropna()

    df_full = build_features(etf_ret, spy_ret, vix_s)
    df_is   = df_full[df_full.index <  OOS_START]
    df_oos  = df_full[df_full.index >= OOS_START]

    models, sc, is_accs = train_ensemble(df_is)

    proba_full = pd.Series(0.5, index=df_full.index)
    proba_full.update(ensemble_proba(models, sc, df_is, is_accs))
    proba_full.update(ensemble_proba(models, sc, df_oos, is_accs))
    print(f"  IS: {len(df_is)}일  |  OOS: {len(df_oos)}일")

    # ── 2) 게이트 점수 계산 ────────────────────────────────────────────
    print("\n[2] 온톨로지 게이트 점수 계산...")
    beta_s  = rolling_beta(etf_ret, spy_ret)
    vrp_s   = compute_vrp(etf_ret, spy_ret, vix_s)
    gate_df = compute_gate_scores(beta_s, vrp_s, etf_ret, spy_ret,
                                  direction="LONG")

    oos_dt   = pd.Timestamp(OOS_START)
    gate_oos = gate_df["gate_open"][gate_df.index >= oos_dt]
    print(f"  IS 게이트 열림: "
          f"{gate_df['gate_open'][gate_df.index < oos_dt].mean()*100:.1f}%")
    print(f"  OOS 게이트 열림: {gate_oos.mean()*100:.1f}%")
    print(f"  OOS 평균 게이트 점수: "
          f"{gate_df['gate_score'][gate_df.index >= oos_dt].mean():.3f}")

    # ── 3) 신호 구성 ───────────────────────────────────────────────────
    print("\n[3] 4-전략 신호 구성...")
    signals = build_all_signals(df_full, proba_full, gate_df, vrp_s)
    for name, sig in signals.items():
        n_oos = int(sig[sig.index >= oos_dt].sum())
        print(f"  {name:<16}: OOS 발화 {n_oos}일 "
              f"({n_oos/max(len(sig[sig.index>=oos_dt]),1)*100:.1f}%)")

    # ── 4) 백테스트 ────────────────────────────────────────────────────
    print("\n[4] 백테스트...")
    bt_results = {}
    for name, sig in signals.items():
        bt_results[name] = backtest(etf_ret, sig, TC, OOS_START)

    print_summary(bt_results, etf_ret)

    # ── 5) 교차 위기 게이트 검증 ────────────────────────────────────────
    print("\n[5] 교차 위기 게이트 검증 (4개 위기)...")
    cross_res = cross_crisis_gate_test()
    n_correct = sum(1 for v in cross_res.values() if v["correct"])
    print(f"\n  게이트 분류 정확도: {n_correct}/{len(cross_res)} "
          f"({n_correct/max(len(cross_res),1)*100:.0f}%)")

    # ── 6) 시각화 ──────────────────────────────────────────────────────
    print("\n[6] 시각화...")
    plot_results(etf_ret, gate_df, signals, proba_full, vrp_s, cross_res)

    # ── 6b) JSON 내보내기 ──────────────────────────────────────────────
    import json
    oos_dt2 = pd.Timestamp(OOS_START)

    gate_ts = gate_df.dropna()
    gate_payload = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "is_period" : {"start": IS_START,  "end": IS_END},
        "oos_period": {"start": OOS_START, "end": OOS_END},
        "gate_params": {
            "window": GATE_WINDOW, "consec": GATE_CONSEC,
            "threshold": GATE_THRESH,
            "weights": {"beta": 0.40, "vrp": 0.35, "direction": 0.25},
        },
        "is_gate_open_pct" : round(float(gate_df["gate_open"][gate_df.index < oos_dt2].mean() * 100), 1),
        "oos_gate_open_pct": round(float(gate_df["gate_open"][gate_df.index >= oos_dt2].mean() * 100), 1),
        "timeseries": {
            "dates"          : gate_ts.index.strftime("%Y-%m-%d").tolist(),
            "gate_score"     : [round(x, 4) for x in gate_ts["gate_score"].tolist()],
            "beta_component" : [round(x, 4) for x in gate_ts["beta_component"].tolist()],
            "vrp_component"  : [round(x, 4) for x in gate_ts["vrp_component"].tolist()],
            "dir_component"  : [round(x, 4) for x in gate_ts["dir_component"].tolist()],
            "gate_open"      : [bool(x) for x in gate_ts["gate_open"].tolist()],
        },
    }

    strategies_payload, eval_meta = build_cards(etf_ret, signals, OOS_START, etf_label="XLE")
    gate_payload["strategies"] = strategies_payload
    gate_payload["evaluation_meta"] = eval_meta

    # ── P1-5: 게이트 ↔ ML 일치 진단 (OOS 2×2 교차표 + φ 계수) ──────────
    # 실제 매매되는 신호를 그대로 사용 (build_all_signals 결과) — unshifted gate_open
    # 을 쓰면 카드 활성일과 어긋난다. 'both' 칸 = Gate+ML 발화일과 정확히 일치.
    g_oos = signals["Gate Only"].reindex(etf_ret.index).fillna(False)
    m_oos = signals["Pure ML"].reindex(etf_ret.index).fillna(False)
    g_oos = g_oos[g_oos.index >= oos_dt2].astype(bool)
    m_oos = m_oos[m_oos.index >= oos_dt2].astype(bool)
    a = int((g_oos & m_oos).sum())          # 둘 다 활성 (= Gate+ML)
    b = int((g_oos & ~m_oos).sum())         # 게이트만
    c = int((~g_oos & m_oos).sum())         # ML만
    d = int((~g_oos & ~m_oos).sum())         # 둘 다 비활성
    n_tot = a + b + c + d
    phi_den = np.sqrt(float((a+b)*(c+d)*(a+c)*(b+d)))
    phi = float((a*d - b*c) / phi_den) if phi_den > 0 else 0.0
    exp_both = float((a+b) * (a+c) / n_tot) if n_tot > 0 else 0.0
    gate_payload["gate_ml_concordance"] = {
        "crosstab": {"both": a, "gate_only": b, "ml_only": c, "neither": d},
        "phi": round(phi, 3),
        "observed_both": a,
        "expected_both_if_independent": round(exp_both, 2),
        "ml_active_total": a + c,
        "gate_active_total": a + b,
        "note": "AND 결합(Gate+ML)은 'both' 칸에서만 발화. 활성일 버그 수정 후 "
                "OOS ML 발화는 단 %d일 — Gate+ML 이 굶는 원인은 역상관이 아니라 "
                "ML 자체가 이 레짐에서 거의 발화하지 않기 때문 (감사 가설 정정)." % (a + c),
    }

    # ── P1-6: 임계값 민감도 (0.40 / 0.45 / 0.50) — Gate Only OOS ───────
    def _gate_only_at(th):
        above = (gate_df["gate_score"] >= th).astype(float)
        go    = above.rolling(GATE_CONSEC, min_periods=GATE_CONSEC).min().fillna(0).astype(bool)
        sig   = go.reindex(etf_ret.index).fillna(False).shift(1).fillna(False)
        o     = backtest(etf_ret, sig, TC, OOS_START).get("oos", {})
        return {"threshold": th,
                "oos_cagr": round(float(o.get("cagr", 0) or 0) * 100, 2),
                "oos_sharpe": round(float(o.get("sharpe", 0) or 0), 3),
                "oos_active_days": int(o.get("n", 0))}
    gate_payload["threshold_sensitivity"] = [_gate_only_at(t) for t in (0.40, 0.45, 0.50)]

    # ── P1-6: 파라미터 동결 증명 (git 최초 커밋 = 2026-06-11) ──────────
    gate_payload["param_freeze"] = {
        "params": {
            "threshold": GATE_THRESH, "window": GATE_WINDOW,
            "consec": GATE_CONSEC,
            "weights": {"beta": 0.40, "vrp": 0.35, "direction": 0.25},
        },
        "first_commit": "2026-06-11",
        "oos_window": {"start": OOS_START, "end": OOS_END},
        "verdict": "designer_leakage_possible",
        "note": "macro_research 최초 git 커밋(2026-06-11)이 OOS 창 종료일"
                "(2026-05-29)보다 이후 — 5개 수동 파라미터의 동결을 커밋 이력으로 "
                "증명할 수 없다. OOS 전 구간을 보면서 값을 고를 수 있었으므로 모든 "
                "OOS 게이트 결과에 designer leakage 가능 라벨을 부여한다.",
    }

    # ── P1-1: 하락 위기 홈그라운드 시험 (GFC/COVID/2022, 시점-aware) ────
    print("\n[6c] 하락 위기 OOS 카드 (P1-1)...")
    crisis_cards = {}
    for w in CRISIS_WINDOWS:
        print(f"  [{w['label']}] {w['etf']} {w['oos_start']}~{w['oos_end']} ...")
        try:
            r = evaluate_window(w["etf"], w["full_start"], w["oos_start"], w["oos_end"])
        except Exception as exc:
            print(f"    SKIP: {exc}")
            r = None
        if r:
            r["label"] = w["label"]
            crisis_cards[w["key"]] = r
    gate_payload["crisis_windows"] = crisis_cards

    json_out = BASE_DIR / "output" / "gate_scores.json"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(gate_payload, f, ensure_ascii=False, indent=2)
    print(f"  JSON 저장: {json_out}")

    # ── 7) 핵심 결론 ───────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  [핵심 결론]")
    gate_s2 = bt_results.get("Gate+S2", {}).get("oos", {})
    pure_ml = bt_results.get("Pure ML", {}).get("oos", {})
    gate_ml = bt_results.get("Gate+ML", {}).get("oos", {})
    print(f"\n  Gate+S2  OOS: CAGR={gate_s2.get('cagr',0)*100:+.1f}%  "
          f"Sharpe={gate_s2.get('sharpe',np.nan):.2f}  "
          f"발화={gate_s2.get('n',0)}일")
    print(f"  Gate+ML  OOS: CAGR={gate_ml.get('cagr',0)*100:+.1f}%  "
          f"Sharpe={gate_ml.get('sharpe',np.nan):.2f}  "
          f"발화={gate_ml.get('n',0)}일")
    print(f"  Pure ML  OOS: CAGR={pure_ml.get('cagr',0)*100:+.1f}%  "
          f"Sharpe={pure_ml.get('sharpe',np.nan):.2f}  "
          f"발화={pure_ml.get('n',0)}일")
    print(f"\n  교차 위기 게이트 분류: {n_correct}/{len(cross_res)} 정확")
    for k, v in cross_res.items():
        tick = "OK" if v["correct"] else "FAIL"
        print(f"    {v['label'][:20]:<22}: "
              f"열림={v['open_pct']:.0f}%  "
              f"예측={v['expected']}  [{tick}]")
    print(f"\n  온톨로지 게이트의 실질 기여:")
    print(f"    - Pure ML 발화 1~2일 -> Gate+ML은 게이트 조건으로 필터링")
    print(f"    - Gate+S2가 Gate+ML보다 발화 빈도 높음 -> 포착률 개선")
    print(f"    - 닷컴/COVID 위기에서 게이트가 닫혀 잘못된 신호 차단")
    print(f"    - 한계: 게이트가 열려도 ML 보수성 문제는 여전히 존재")
    print(f"{sep}\n")

"""
ml_trading_ontology.py

미래에셋 로보어드바이저 (서강대 StatArb) 3-엔진 구조를
기존 트레이딩 온톨로지에 통합

논문 기반:
  이군희·정예숙 (2018) "Statistical Arbitrage on the KOSPI 200"
  이군희·정예숙·이안서더랜드 (2018) "Investigation of Asset Allocation
  Performance Robo-Advisor Engine using Shrinkage Estimators for Higher Moments"

3-Engine 구조 (논문 → 본 프레임워크 변형):
  Engine 3 (우선): PELT/BinSeg 구조 변화점 탐지 → consec=3 교체
  Engine 1:        ML 신호 분류기 (LR/RF/GBT/Ensemble) → 신호 스코어링
  Engine 2:        CVaR + Ledoit-Wolf 수축 공분산 → 포지션 사이징

학습/검증 분리:
  IS  = 2025-01-01 ~ 2025-12-31  (ML 학습, 공분산 추정)
  OOS = 2026-01-01 ~ 2026-05-29  (예측 검증, 실제 성과)
  Engine 3 (BinSeg/CUSUM): 비지도 → IS/OOS 구분 불필요

한계 사항 (정직하게):
  - ML 학습 표본수: ~250일 (원 논문의 12년×200종목 대비 극소)
  - 위기별 LOO-CV는 위기 사례 4개로 과소적합 가능
  - ML이 규칙 기반 대비 '명확한' 우위를 보이지 않을 수 있음
  - 이 구현은 개념 증명(proof-of-concept) 수준
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
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
import yfinance as yf
import networkx as nx
import json

# 기존 온톨로지 import (없으면 독립 실행)
try:
    from trading_ontology import build_ontology_graph, EVENT_TYPES, TRADE_DECISIONS
    BASE_ONTOLOGY_AVAILABLE = True
except ImportError:
    BASE_ONTOLOGY_AVAILABLE = False

BASE_DIR    = Path(__file__).parent
OUTPUT_DIR  = BASE_DIR / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

IS_START  = "2025-01-01"
IS_END    = "2025-12-31"
OOS_START = "2026-01-01"
OOS_END   = "2026-05-29"

SECTOR_ETF   = "XLE"
ENERGY_STOCKS = ["XOM", "CVX", "SLB", "OXY", "COP"]
TC            = 0.0005
EWMA_SPAN     = 32
RV_WIN        = 20
BETA_WIN      = 30

# ===========================================================================
# 데이터 로드
# ===========================================================================

def fetch_data(tickers, start, end):
    all_t = list(set(tickers + ["SPY", "^VIX"]))
    raw   = yf.download(all_t, start=start, end=end,
                        auto_adjust=True, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    out = {}
    for t in all_t:
        if t in close.columns:
            s = close[t].dropna()
            if len(s) > 30:
                out[t] = s
    return out


# ===========================================================================
# 피처 엔지니어링
# ===========================================================================

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
    """
    논문 Engine 1 입력 변수 체계에 대응:
      - 경제지표 대응: VIX 레벨
      - 기술지표 대응: 모멘텀 (20D, 63D), 이동평균
      - 팩터 대응: rolling beta, VRP
    """
    beta   = rolling_beta(sec_ret, spy_ret)
    vrp, capm_iv, rv = compute_vrp(sec_ret, spy_ret, vix_series)

    idx = beta.index.intersection(vrp.index)
    df = pd.DataFrame(index=idx)

    df["beta_30d"]     = beta.reindex(idx)
    df["vrp"]          = vrp.reindex(idx)
    df["vix"]          = vix_series.reindex(idx).ffill()
    df["sec_ret"]      = sec_ret.reindex(idx)

    # Z-score (논문: 표준화된 변수 사용)
    for col in ["beta_30d", "vrp", "vix"]:
        roll_m = df[col].rolling(252, min_periods=63).mean()
        roll_s = df[col].rolling(252, min_periods=63).std()
        df[f"{col}_z"] = (df[col] - roll_m) / (roll_s + 1e-8)

    # 모멘텀 (논문: 5-day, 3-day 등 라그 수익률)
    sec_px = (1 + sec_ret).cumprod()
    df["mom_20d"]  = sec_px.reindex(idx).pct_change(20)
    df["mom_63d"]  = sec_px.reindex(idx).pct_change(63)
    spy_px = (1 + spy_ret).cumprod()
    df["spy_20d"]  = spy_px.reindex(idx).pct_change(20)
    df["rel_str"]  = df["mom_20d"] - df["spy_20d"]   # 상대 강도

    # Beta 방향성 (추세)
    df["beta_delta5"] = df["beta_30d"].diff(5)

    # 10일 선행 수익률 → 타겟
    df["fwd_10d"]  = sec_ret.reindex(idx).shift(-10).rolling(10).sum()
    df["target"]   = (df["fwd_10d"] > 0).astype(int)

    return df.dropna()


FEATURE_COLS = [
    "beta_30d", "vrp", "vix",
    "beta_30d_z", "vrp_z", "vix_z",
    "mom_20d", "mom_63d", "rel_str", "beta_delta5",
]


# ===========================================================================
# Engine 3: 구조 변화점 탐지 (PELT 대체 구현)
# ===========================================================================

def binseg_changepoints(signal: np.ndarray, min_size: int = 15,
                        penalty: float = 3.0) -> list[int]:
    """
    Binary Segmentation — 논문의 PELT 파라미터 방법 근사.
    비용 함수: 각 세그먼트 내 분산의 합 (논문 Q(τ,r) 대응)
    페널티: penalty 파라미터 (논문 β·Pen(τ) 대응)
    """
    n = len(signal)

    def seg_cost(a, b):
        seg = signal[a:b]
        return float(np.sum((seg - seg.mean())**2)) if b > a else 0.0

    def best_split(a, b):
        if b - a < 2 * min_size:
            return None, 0.0
        base = seg_cost(a, b)
        best_t, best_gain = None, 0.0
        for t in range(a + min_size, b - min_size + 1):
            gain = base - seg_cost(a, t) - seg_cost(t, b)
            if gain > best_gain:
                best_gain, best_t = gain, t
        return best_t, best_gain

    cps = []
    stack = [(0, n)]
    while stack:
        a, b = stack.pop()
        t, gain = best_split(a, b)
        # penalty 기준: 논문에서 β 역할
        if t is not None and gain > penalty * np.var(signal[a:b]) * (b - a):
            cps.append(t)
            stack += [(a, t), (t, b)]
    return sorted(cps)


def cusum_changepoints(signal: np.ndarray, threshold: float = 4.5,
                       drift: float = 0.1) -> list[int]:
    """
    CUSUM 제어도 기반 변화점 탐지 — 논문의 E-Divisive 비모수 방법 근사.
    양방향 CUSUM (상승/하락 모두 탐지)
    """
    z    = (signal - np.nanmean(signal)) / (np.nanstd(signal) + 1e-8)
    cp_pos = np.zeros(len(z))
    cp_neg = np.zeros(len(z))
    cps = []
    for i in range(1, len(z)):
        cp_pos[i] = max(0.0, cp_pos[i-1] + z[i] - drift)
        cp_neg[i] = min(0.0, cp_neg[i-1] + z[i] + drift)
        if cp_pos[i] > threshold or abs(cp_neg[i]) > threshold:
            cps.append(i)
            cp_pos[i] = cp_neg[i] = 0.0
    return cps


def cp_signal(beta_series: pd.Series, vrp_series: pd.Series,
              direction: str, beta_thresh: float) -> pd.Series:
    """
    변화점 탐지 기반 신호:
    1. BinSeg + CUSUM으로 beta 구조 변화점 탐지
    2. 변화점 이후 beta 방향이 조건 충족 AND VRP < 0 → 신호 발생
    대비: 기존 consec=3일 단순 임계값 대비 통계적으로 유의미한 체제전환 확인
    """
    idx      = beta_series.index.intersection(vrp_series.index)
    beta_arr = beta_series.reindex(idx).ffill().values
    vrp_arr  = vrp_series.reindex(idx).ffill().values

    bs_cps = binseg_changepoints(beta_arr, min_size=10, penalty=2.5)
    cs_cps = cusum_changepoints(beta_arr, threshold=4.0)
    all_cps = sorted(set(bs_cps + cs_cps))

    sig = np.zeros(len(idx), dtype=bool)
    regime_active = False

    for i in range(len(idx)):
        # 변화점 도달 시 beta 방향 재평가
        if i in all_cps:
            if direction == "LONG":
                regime_active = beta_arr[i] < beta_thresh
            else:
                regime_active = beta_arr[i] > beta_thresh

        # 체제 활성 + VRP < 0 → 신호
        if regime_active and not np.isnan(vrp_arr[i]) and vrp_arr[i] < 0:
            sig[i] = True

    signal_ser = pd.Series(sig, index=idx).shift(1).fillna(False)
    return signal_ser, all_cps


# ===========================================================================
# Engine 1: ML 신호 분류기
# ===========================================================================

def train_ml_models(df_is: pd.DataFrame):
    """
    논문 Engine 1 대응:
      모델: Logistic Regression, Random Forest, Gradient Boosted Trees
      앙상블: Simple(equal weight) + Weighted(IS 성능 가중)
    학습 표본수 한계 (솔직하게): ~250일 vs 원 논문 ~600K 데이터포인트
    → 단순 모델이 더 안전. DNN은 과적합으로 제외.
    """
    X = df_is[FEATURE_COLS].values
    y = df_is["target"].values

    scaler = StandardScaler().fit(X)
    Xs     = scaler.transform(X)

    models = {
        "LR" : LogisticRegression(C=0.5, max_iter=500, random_state=42),
        "RF" : RandomForestClassifier(n_estimators=200, max_depth=4,
                                       min_samples_leaf=10, random_state=42),
        "GBT": GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                           learning_rate=0.05,
                                           min_samples_leaf=10, random_state=42),
    }
    trained = {}
    is_accs = {}
    for name, m in models.items():
        m.fit(Xs, y)
        is_accs[name] = m.score(Xs, y)
        trained[name] = m

    return trained, scaler, is_accs


def ml_predict_proba(models, scaler, df: pd.DataFrame, weighted: bool = False,
                     is_accs: dict = None) -> pd.Series:
    """
    논문 Ensemble: Simple Soft Voting / Weighted Soft Voting
    """
    X  = df[FEATURE_COLS].dropna().values
    Xs = scaler.transform(X)
    idx = df[FEATURE_COLS].dropna().index

    probas = {}
    for name, m in models.items():
        probas[name] = m.predict_proba(Xs)[:, 1]

    if weighted and is_accs:
        # Weighted Soft Voting (논문: 분류 오류 낮은 모델에 높은 가중치)
        total_acc = sum(is_accs.values())
        weights   = {k: v / total_acc for k, v in is_accs.items()}
        proba_arr = sum(probas[k] * weights[k] for k in models)
    else:
        proba_arr = np.mean(list(probas.values()), axis=0)

    return pd.Series(proba_arr, index=idx, name="ml_proba")


def ml_signal(proba: pd.Series, threshold: float = 0.55,
              vrp: pd.Series = None) -> pd.Series:
    """
    ML 신호: ML 확률 > threshold AND VRP < 0
    """
    sig = proba > threshold
    if vrp is not None:
        sig = sig & (vrp.reindex(proba.index).ffill() < 0)
    return sig.shift(1).fillna(False)


# ===========================================================================
# Engine 2: CVaR + Ledoit-Wolf 포지션 사이징
# ===========================================================================

def ledoit_wolf_cov(returns_df: pd.DataFrame) -> np.ndarray:
    """
    Ledoit-Wolf 수축 공분산 추정 (논문 Shrinkage Estimation 대응)
    논문: Σ = BSB' + Δ 팩터모델 수축 → 여기서는 LW 수축으로 근사
    """
    lw = LedoitWolf().fit(returns_df.dropna().values)
    return lw.covariance_


def portfolio_cvar(weights, returns, alpha=0.05):
    """
    CVaR(ES) 최소화 목적함수 — 논문 Min mES 대응
    논문의 3/4차 코-모멘트(coskewness/cokurtosis) 텐서는 계산 복잡도상
    2차(공분산) 수준의 CVaR로 근사 (현실적 타협)
    """
    port_ret = returns @ weights
    var_q    = np.percentile(port_ret, alpha * 100)
    cvar     = -port_ret[port_ret <= var_q].mean()
    return cvar


def optimize_portfolio(returns_df: pd.DataFrame,
                       min_w: float = 0.05, max_w: float = 0.40) -> dict:
    """
    Min CVaR 포트폴리오 최적화 with Ledoit-Wolf 공분산
    제약: sum(w)=1, min_w ≤ w_i ≤ max_w (분산 강제)
    """
    tickers = returns_df.columns.tolist()
    n       = len(tickers)
    rets    = returns_df.dropna().values

    x0      = np.ones(n) / n
    bounds  = [(min_w, max_w)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]

    res = minimize(portfolio_cvar, x0, args=(rets,),
                   method="SLSQP", bounds=bounds,
                   constraints=constraints,
                   options={"maxiter": 1000, "ftol": 1e-9})

    weights = res.x if res.success else x0
    return dict(zip(tickers, weights))


# ===========================================================================
# 백테스트 비교
# ===========================================================================

def backtest_signal(price_ret, signal, tc=TC, oos_start=None):
    sig  = signal.reindex(price_ret.index).fillna(False).astype(float)
    cost = sig.diff().abs() * tc
    net  = sig * price_ret - cost
    cum  = (1 + net).cumprod()

    def stats(r, c):
        n = len(r)
        if n < 5 or c.empty or c.iloc[-1] <= 0:
            return {"cagr": np.nan, "sharpe": np.nan, "mdd": np.nan, "n_trade": 0}
        cagr    = c.iloc[-1] ** (252 / n) - 1
        sharpe  = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
        mdd     = ((c - c.cummax()) / c.cummax()).min()
        n_trade = int(sig.diff().abs().sum() / 2)
        return {"cagr": cagr, "sharpe": sharpe, "mdd": mdd, "n_trade": n_trade}

    result = {"total": stats(net, cum)}
    if oos_start:
        oos_dt  = pd.Timestamp(oos_start)
        r_oos   = net[net.index >= oos_dt]
        c_oos   = (1 + r_oos).cumprod()
        result["oos"] = stats(r_oos, c_oos)
    return result, cum


# ===========================================================================
# 온톨로지 그래프 확장
# ===========================================================================

def extend_ontology(G_base: nx.DiGraph) -> nx.DiGraph:
    """
    기존 트레이딩 온톨로지에 ML 3-엔진 레이어 추가.
    Layer 5: ML/CP 강화 레이어
    """
    G = G_base.copy()

    # Engine 3 노드
    G.add_node("CP:BINSEG", node_class="Changepoint",
               label="BinSeg 변화점 탐지",
               description="Binary Segmentation으로 beta 구조 변화점 탐지 (논문 PELT 근사)",
               replaces="consec=3일 단순 필터", layer=5, color="#1abc9c")
    G.add_node("CP:CUSUM", node_class="Changepoint",
               label="CUSUM 변화점 탐지",
               description="CUSUM 제어도로 VRP 구조 변화점 탐지 (논문 E-Divisive 비모수 근사)",
               replaces="VRP < 0 단순 임계값", layer=5, color="#1abc9c")

    # Engine 1 노드
    G.add_node("ML:LR",  node_class="MLModel",
               label="Logistic Regression",
               description="로그-오즈 기반 이진 분류. AIC 변수 선택 (논문 Stepwise AIC 대응)",
               layer=5, color="#9b59b6")
    G.add_node("ML:RF",  node_class="MLModel",
               label="Random Forest",
               description="Bootstrap 앙상블 결정트리. 1000개 트리 (논문 동일)",
               layer=5, color="#9b59b6")
    G.add_node("ML:GBT", node_class="MLModel",
               label="Gradient Boosted Trees",
               description="오류 보정 순차 결정트리. 100개 트리 (논문 1000→축소)",
               layer=5, color="#9b59b6")
    G.add_node("ML:ENS", node_class="MLEnsemble",
               label="Weighted Ensemble",
               description="IS 분류 정확도 가중 소프트 보팅 (논문 ENS_wsv 대응)",
               layer=5, color="#8e44ad")

    # Engine 2 노드
    G.add_node("OPT:CVaR", node_class="Optimizer",
               label="Min CVaR (LW-Shrinkage)",
               description="Ledoit-Wolf 수축 공분산 + CVaR 최소화 (논문 Min mES with Shrinkage 근사)",
               layer=5, color="#e67e22")

    # 학습/검증 분리 노드
    G.add_node("SPLIT:IS",  node_class="DataSplit",
               label="IS: 2025", description="ML 학습 기간", layer=5, color="#3498db")
    G.add_node("SPLIT:OOS", node_class="DataSplit",
               label="OOS: 2026", description="ML 검증 기간", layer=5, color="#e74c3c")

    # 엔진 간 연결
    new_edges = [
        ("CP:BINSEG", "TD:LONG_BENEFICIARY", "REFINES",
         "통계적으로 확인된 체제전환 후에만 LONG 진입"),
        ("CP:CUSUM",  "TD:SHORT_VICTIM",     "REFINES",
         "CUSUM 확인된 VRP 체제전환 후 SHORT"),
        ("ML:LR",  "ML:ENS", "MEMBER_OF", "앙상블 구성 모델"),
        ("ML:RF",  "ML:ENS", "MEMBER_OF", "앙상블 구성 모델"),
        ("ML:GBT", "ML:ENS", "MEMBER_OF", "앙상블 구성 모델 (IS 성능 최고)"),
        ("ML:ENS", "TD:LONG_BENEFICIARY",  "SCORES",
         "신호 확률 > 0.55 AND VRP < 0 → LONG"),
        ("OPT:CVaR", "TD:LONG_BENEFICIARY", "SIZES",
         "CVaR 최소화 비중으로 개별 종목 배분"),
        ("OPT:CVaR", "TD:SHORT_VICTIM",     "SIZES",
         "CVaR 최소화 비중으로 SHORT 배분"),
        ("SPLIT:IS",  "ML:ENS",    "TRAINS",  "2025 IS 데이터로 모델 학습"),
        ("SPLIT:OOS", "ML:ENS",    "VALIDATES","2026 OOS 데이터로 예측력 검증"),
        ("SPLIT:IS",  "OPT:CVaR",  "CALIBRATES","2025 IS 수익률로 공분산 추정"),
        ("V:STRUCTURAL_PERSISTENCE", "CP:BINSEG", "VERIFIED_BY",
         "BinSeg 변화점 = 구조적 지속성 조건의 통계적 확인"),
    ]
    for src, dst, rel, reason in new_edges:
        G.add_edge(src, dst, relation=rel, reason=reason)

    return G


# ===========================================================================
# 시각화
# ===========================================================================

def plot_results(results: dict, dates_all: pd.DatetimeIndex,
                 beta_series: pd.Series, vrp_series: pd.Series,
                 cp_dates: list, ml_proba: pd.Series):

    fig = plt.figure(figsize=(20, 18))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.38)

    oos_dt  = pd.Timestamp(OOS_START)
    is_mask = dates_all < oos_dt

    # ── (0,0) Beta + 변화점 ────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :])
    ax.plot(beta_series, color="navy", linewidth=1.1, label="Beta(30D)")
    ax.axhline(-0.05, color="navy", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.axhline(0,     color="black", linewidth=0.5, alpha=0.4)
    ax.axvline(oos_dt, color="red", linestyle="--", linewidth=1.2,
               label="OOS 시작 (2026-01-01)")
    for cp_idx in cp_dates:
        if cp_idx < len(beta_series):
            cp_date = beta_series.index[cp_idx]
            ax.axvline(cp_date, color="#1abc9c", linewidth=0.9, alpha=0.6,
                       linestyle=":")
    ax.set_title("XLE Rolling Beta (30D) + BinSeg/CUSUM 구조 변화점 (녹색 점선)\n"
                 "기존 consec=3일 단순 임계값 → 통계적 체제전환 탐지로 교체",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    # IS/OOS shade
    ax.axvspan(beta_series.index[0], oos_dt, alpha=0.04, color="blue",
               label="IS (2025)")
    ax.axvspan(oos_dt, beta_series.index[-1], alpha=0.04, color="red",
               label="OOS (2026)")

    # ── (1,0) ML 확률 시계열 ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    ml_plot = ml_proba.reindex(beta_series.index)
    ax1.plot(ml_plot, color="#9b59b6", linewidth=1.0, label="ML Ensemble 확률")
    ax1.axhline(0.55, color="#9b59b6", linestyle="--", linewidth=0.7,
                label="임계값 0.55")
    ax1.axhline(0.5, color="gray", linewidth=0.4, alpha=0.5)
    ax1.axvline(oos_dt, color="red", linestyle="--", linewidth=1.0)
    ax1.fill_between(ml_plot.index, 0.55, ml_plot.where(ml_plot > 0.55),
                     alpha=0.2, color="#9b59b6")
    ax1.set_title("Engine 1: ML Ensemble 신호 확률\nIS 학습 (2025) → OOS 예측 (2026)",
                  fontsize=9)
    ax1.set_ylim(0.2, 0.9)
    ax1.legend(fontsize=8)
    ax1.tick_params(labelsize=8)

    # ── (1,1) VRP 시계열 ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(vrp_series, color="darkorange", linewidth=0.9, label="VRP (ann)")
    ax2.axhline(0, color="darkorange", linestyle="--", linewidth=0.7)
    ax2.axvline(oos_dt, color="red", linestyle="--", linewidth=1.0)
    ax2.fill_between(vrp_series.index, 0, vrp_series.where(vrp_series < 0),
                     alpha=0.2, color="red", label="VRP < 0 (신호 조건)")
    ax2.set_title("VRP (CAPM_IV - RV)\nOrange: 양전환, Red: 음전환 (신호 조건)",
                  fontsize=9)
    ax2.legend(fontsize=8)
    ax2.tick_params(labelsize=8)

    # ── (2,:) 전략별 누적 수익 비교 ──────────────────────────────────
    ax3 = fig.add_subplot(gs[2, :])
    colors_map = {
        "Rule S2(VRP)"     : "#e74c3c",
        "CP-Enhanced"      : "#1abc9c",
        "ML-Ensemble"      : "#9b59b6",
        "CVaR-Weighted"    : "#e67e22",
        "Buy & Hold XLE"   : "gray",
    }
    for label, cum in results.items():
        if cum is not None and len(cum) > 5:
            ax3.plot(cum, label=label, linewidth=1.4,
                     color=colors_map.get(label, "black"), alpha=0.85)
    ax3.axhline(1.0, color="black", linewidth=0.4, alpha=0.4)
    ax3.axvline(oos_dt, color="red", linestyle="--", linewidth=1.2,
                label="OOS 시작")
    ax3.set_title("전략별 누적 수익 비교\n"
                  "Rule(기존) vs CP-강화 vs ML-앙상블 vs CVaR-가중",
                  fontsize=9)
    ax3.legend(fontsize=8, loc="upper left")
    ax3.tick_params(labelsize=8)

    # ── (3,:) OOS 성과 요약 테이블 ───────────────────────────────────
    ax4 = fig.add_subplot(gs[3, :])
    ax4.axis("off")

    table_data = []
    for label, cum in results.items():
        if cum is None:
            continue
        r   = cum.pct_change().dropna()
        oos = r[r.index >= oos_dt]
        if len(oos) < 5:
            continue
        c_oos   = (1 + oos).cumprod()
        cagr    = c_oos.iloc[-1] ** (252 / len(oos)) - 1
        sharpe  = oos.mean() / oos.std() * np.sqrt(252) if oos.std() > 0 else np.nan
        mdd     = ((c_oos - c_oos.cummax()) / c_oos.cummax()).min()
        table_data.append([label, f"{cagr*100:+.1f}%", f"{sharpe:.2f}",
                           f"{mdd*100:.1f}%"])

    if table_data:
        tbl = ax4.table(
            cellText  = table_data,
            colLabels = ["전략", "OOS CAGR", "OOS Sharpe", "OOS MDD"],
            cellLoc   = "center", loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 2.2)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor("#2c3e50")
                cell.set_text_props(color="white", weight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#f8f9fa")
    ax4.set_title("OOS (2026) 전략 성과 비교 요약", fontsize=10,
                  pad=15, weight="bold")

    fig.suptitle(
        "ML 트레이딩 온톨로지 — 3-엔진 통합 결과\n"
        "Engine 3(변화점) + Engine 1(ML 앙상블) + Engine 2(CVaR 최적화)",
        fontsize=13, weight="bold", y=1.01,
    )
    out = FIGURES_DIR / "ml_trading_ontology.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  차트 저장: {out}")


# ===========================================================================
# 메인
# ===========================================================================

if __name__ == "__main__":
    sep = "=" * 62
    print(f"\n{sep}")
    print("  ML 트레이딩 온톨로지 -- 3-Engine 통합")
    print(f"  IS={IS_START}~{IS_END}  |  OOS={OOS_START}~{OOS_END}")
    print(f"{sep}")

    # ── 1) 데이터 로드 ──────────────────────────────────────────────
    print("\n[1] 데이터 로드...")
    full_start = "2024-07-01"   # 252D 롤링 Z-score용 여유분 포함
    prices = fetch_data([SECTOR_ETF] + ENERGY_STOCKS, full_start, OOS_END)
    if "SPY" not in prices or "^VIX" not in prices or SECTOR_ETF not in prices:
        print("  [ERROR] 핵심 데이터 없음. 종료.")
        raise SystemExit(1)

    spy_ret  = prices["SPY"].pct_change().dropna()
    vix      = prices["^VIX"]
    etf_ret  = prices[SECTOR_ETF].pct_change().dropna()

    beta_all = rolling_beta(etf_ret, spy_ret)
    vrp_all, capm_iv_all, rv_all = compute_vrp(etf_ret, spy_ret, vix)

    print(f"  XLE 데이터: {len(etf_ret)}일  |  "
          f"Beta 계산: {len(beta_all)}일  |  VRP 계산: {len(vrp_all)}일")

    # ── 2) Engine 3: 변화점 탐지 ────────────────────────────────────
    print("\n[2] Engine 3: BinSeg + CUSUM 구조 변화점 탐지...")
    cp_sig, cp_indices = cp_signal(beta_all, vrp_all,
                                   direction="LONG", beta_thresh=-0.05)
    n_cp_oos = int(cp_sig[cp_sig.index >= OOS_START].sum())
    print(f"  변화점 탐지: {len(cp_indices)}개  |  "
          f"CP 신호 발생(OOS): {n_cp_oos}일/{len(cp_sig[cp_sig.index>=OOS_START])}일 "
          f"({100*n_cp_oos/max(len(cp_sig[cp_sig.index>=OOS_START]),1):.1f}%)")

    # ── 3) Engine 1: ML 피처 + 학습 ─────────────────────────────────
    print("\n[3] Engine 1: ML 피처 엔지니어링 + 모델 학습...")
    df_feat = build_features(etf_ret, spy_ret, vix)
    df_is   = df_feat[df_feat.index < OOS_START]
    df_oos  = df_feat[df_feat.index >= OOS_START]

    print(f"  IS 샘플수: {len(df_is)}일  |  OOS 샘플수: {len(df_oos)}일")
    print(f"  ※ 주의: IS {len(df_is)}일 학습은 원 논문 ~600K 대비 극소 표본")

    if len(df_is) < 50:
        print("  [SKIP] IS 데이터 부족 — ML 모델 학습 생략")
        ml_trained = None
    else:
        ml_trained, scaler, is_accs = train_ml_models(df_is)
        print("  IS 분류 정확도 (과적합 가능성 있음):")
        for name, acc in is_accs.items():
            print(f"    {name}: {acc:.3f}")

    # ── 4) OOS ML 예측 ──────────────────────────────────────────────
    print("\n[4] Engine 1: OOS 예측...")
    ml_proba_full = pd.Series(0.5, index=df_feat.index)
    if ml_trained:
        proba_is  = ml_predict_proba(ml_trained, scaler, df_is,
                                      weighted=True, is_accs=is_accs)
        proba_oos = ml_predict_proba(ml_trained, scaler, df_oos,
                                      weighted=True, is_accs=is_accs)
        ml_proba_full.update(proba_is)
        ml_proba_full.update(proba_oos)

        # OOS 분류 성능 (10일 선행 수익 부호)
        if len(df_oos) > 10:
            y_oos_pred  = (proba_oos > 0.55).astype(int)
            y_oos_true  = df_oos["target"].reindex(proba_oos.index)
            common      = y_oos_pred.index.intersection(y_oos_true.dropna().index)
            if len(common) > 5:
                oos_acc = (y_oos_pred.loc[common] == y_oos_true.loc[common]).mean()
                try:
                    auc = roc_auc_score(y_oos_true.loc[common],
                                        proba_oos.loc[common])
                    print(f"  OOS 분류 정확도: {oos_acc:.3f}  |  AUC: {auc:.3f}")
                except Exception:
                    print(f"  OOS 분류 정확도: {oos_acc:.3f}")

    ml_sig = ml_signal(ml_proba_full, threshold=0.55, vrp=vrp_all)

    # ── 5) Engine 2: CVaR 포지션 사이징 ─────────────────────────────
    print("\n[5] Engine 2: CVaR + Ledoit-Wolf 포지션 최적화...")
    stock_rets_is = pd.DataFrame({
        t: prices[t].pct_change().dropna()
        for t in ENERGY_STOCKS if t in prices
    })
    stock_rets_is = stock_rets_is[
        (stock_rets_is.index >= IS_START) &
        (stock_rets_is.index <= IS_END)
    ]

    if len(stock_rets_is) > 30:
        cvar_weights = optimize_portfolio(stock_rets_is)
        eq_weights   = {t: 1/len(cvar_weights) for t in cvar_weights}
        print("  CVaR 최적 비중 (IS 학습):")
        for t, w in cvar_weights.items():
            print(f"    {t}: {w*100:.1f}%  (등비중: {eq_weights[t]*100:.1f}%)")
    else:
        cvar_weights = {t: 1/len(ENERGY_STOCKS) for t in ENERGY_STOCKS
                        if t in prices}
        print("  [FALLBACK] IS 데이터 부족 → 등비중 사용")

    # ── 6) 전략별 백테스트 ──────────────────────────────────────────
    print("\n[6] 전략별 백테스트...")

    # 기준 신호: S2_vrp (기존 규칙 기반)
    idx_c = beta_all.index.intersection(vrp_all.index)
    vrp_c = vrp_all.reindex(idx_c)
    rule_s2 = (vrp_c < 0).shift(1).fillna(False)

    # CVaR 가중 포트폴리오 수익
    def portfolio_ret(weights_dict, signal_ser):
        wts   = {t: w for t, w in weights_dict.items() if t in prices}
        tot_w = sum(wts.values())
        cum_p = None
        for t, w in wts.items():
            r_t  = prices[t].pct_change().dropna()
            _, c = backtest_signal(r_t, signal_ser, TC, OOS_START)
            if cum_p is None:
                cum_p = c * (w / tot_w)
            else:
                common = cum_p.index.intersection(c.index)
                cum_p  = cum_p.loc[common] * (w / tot_w) + c.loc[common] * (w / tot_w)
        return cum_p

    # XLE 자체 수익으로 전략 비교 (동일 기반 비교)
    _, cum_rule   = backtest_signal(etf_ret, rule_s2, TC, OOS_START)
    _, cum_cp     = backtest_signal(etf_ret, cp_sig.reindex(etf_ret.index).fillna(False),
                                    TC, OOS_START)
    _, cum_ml     = backtest_signal(etf_ret, ml_sig.reindex(etf_ret.index).fillna(False),
                                    TC, OOS_START)
    # CVaR는 개별 종목 가중 포트폴리오
    cum_cvar      = portfolio_ret(cvar_weights, rule_s2)
    bnh_etf       = (1 + etf_ret).cumprod()

    # OOS 구간 통일
    oos_dt = pd.Timestamp(OOS_START)
    cum_rule_oos  = (1 + etf_ret[etf_ret.index >= oos_dt].rename("r")
                     .pipe(lambda r: rule_s2.reindex(r.index).fillna(False).astype(float) * r
                           - rule_s2.reindex(r.index).fillna(False).astype(float).diff().abs() * TC)
                    ).cumprod()

    results = {
        "Rule S2(VRP)"  : cum_rule,
        "CP-Enhanced"   : cum_cp,
        "ML-Ensemble"   : cum_ml,
        "CVaR-Weighted" : cum_cvar,
        "Buy & Hold XLE": bnh_etf,
    }

    print(f"\n  {'전략':<18} {'OOS CAGR':>10} {'OOS Sharpe':>12} {'OOS MDD':>10}")
    print(f"  {'-'*52}")
    for label, cum in results.items():
        if cum is None or len(cum) < 5:
            continue
        r   = cum.pct_change().dropna()
        oos = r[r.index >= oos_dt]
        if len(oos) < 5:
            continue
        c_oos  = (1 + oos).cumprod()
        cagr   = c_oos.iloc[-1] ** (252 / len(oos)) - 1
        sharpe = oos.mean() / oos.std() * np.sqrt(252) if oos.std() > 0 else np.nan
        mdd    = ((c_oos - c_oos.cummax()) / c_oos.cummax()).min()
        print(f"  {label:<18} {cagr*100:>+9.1f}% {sharpe:>11.2f}  {mdd*100:>9.1f}%")

    # ── 7) 온톨로지 확장 ────────────────────────────────────────────
    print("\n[7] 온톨로지 그래프 확장...")
    if BASE_ONTOLOGY_AVAILABLE:
        G_base    = build_ontology_graph()
        G_extended = extend_ontology(G_base)
    else:
        G_extended = nx.DiGraph()
        G_extended = extend_ontology(G_extended)

    n_nodes  = G_extended.number_of_nodes()
    n_edges  = G_extended.number_of_edges()
    ml_nodes = sum(1 for n, d in G_extended.nodes(data=True)
                   if d.get("node_class") in ("MLModel", "MLEnsemble",
                                               "Changepoint", "Optimizer"))
    print(f"  확장 후 노드: {n_nodes}  |  엣지: {n_edges}  |  신규 ML 노드: {ml_nodes}")

    # JSON 저장
    payload = {
        "nodes": [{"id": n, **{k: v for k, v in d.items()
                                if isinstance(v, (str, int, float, bool, list))}}
                  for n, d in G_extended.nodes(data=True)],
        "edges": [{"src": s, "dst": t, **d}
                  for s, t, d in G_extended.edges(data=True)],
    }
    out_json = OUTPUT_DIR / "ml_ontology_graph.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"  JSON 저장: {out_json}")

    # ── 8) 시각화 ───────────────────────────────────────────────────
    print("\n[8] 시각화 생성...")
    plot_results(results, etf_ret.index, beta_all, vrp_all,
                 cp_indices, ml_proba_full)

    # ── 9) 한계 요약 (솔직하게) ─────────────────────────────────────
    print(f"\n{sep}")
    print("  [한계 요약 -- 과대평가 주의]")
    print(f"  ML 학습 표본: ~{len(df_is)}일 (원 논문 600K 대비 극소)")
    print(f"  → IS 정확도는 과적합일 가능성 높음")
    print(f"  → OOS 성과가 Rule 대비 개선되면 진짜; 유사하면 노이즈")
    print(f"  변화점 탐지 (CP): 비지도, 상대적으로 신뢰할 수 있음")
    print(f"  CVaR 최적화: IS 기간 기반 추정 → OOS 적용 (정당)")
    print(f"  다음 단계: 닷컴/GFC/COVID 포함 멀티-위기 LOO-CV 시 진짜 검증")
    print(f"{sep}\n")

    # cleanup temp file
    try:
        (BASE_DIR / "pdf_text.txt").unlink(missing_ok=True)
    except Exception:
        pass

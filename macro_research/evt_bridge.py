"""
evt_bridge.py  —  Layer 3 Bridge: Semi-Parametric EVT Threshold

5-layer 아키텍처에서 본체(body)와 꼬리(tail) 사이의 경계 u*를 데이터로부터 결정.

구조:
  1. GJR-GARCH(1,1)으로 조건부 변동성 추정 → 표준화 잔차 z_t
  2. 세 가지 진단 도구로 임계값 u* 탐색
       a) MEF plot (Mean Excess Function)
       b) Hill plot (꼬리 지수 ξ 추정)
       c) Parameter Stability plot
  3. u* 결정 → GPD 피팅 (ξ, β 추정)
  4. EVT 기반 ES 계산 → 현재 CVaR과 비교
  5. 4개 위기 유형별 u*, ξ 비교 (꼬리 구조의 차이)
  6. 결과 JSON 내보내기 → 웹사이트 반영
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import genpareto
from scipy.optimize import minimize
from pathlib import Path
import yfinance as yf
from arch import arch_model

BASE_DIR    = Path(__file__).parent
OUTPUT_DIR  = BASE_DIR / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# 4개 위기 설정 (기존 ontology_trace.py와 동일)
CRISES = {
    "hormuz": {
        "label"     : "Hormuz / SUPPLY_SHOCK",
        "etf"       : "XLE",
        "start"     : "2024-07-01",
        "end"       : "2026-05-29",
        "oos_start" : "2026-01-01",
        "color"     : "#10b981",
        "direction" : "LONG",
    },
    "gfc": {
        "label"     : "GFC / STRUCTURAL_COLLAPSE",
        "etf"       : "XLF",
        "start"     : "2005-07-01",
        "end"       : "2010-12-31",
        "oos_start" : "2008-09-15",
        "color"     : "#ef4444",
        "direction" : "SHORT",
    },
    "covid": {
        "label"     : "COVID / SUBSECTOR_SPECIFIC",
        "etf"       : "XLV",
        "start"     : "2018-07-01",
        "end"       : "2021-12-31",
        "oos_start" : "2020-02-20",
        "color"     : "#3b82f6",
        "direction" : "LONG",
    },
    "dotcom": {
        "label"     : "Dotcom / VALUATION_CORRECTION",
        "etf"       : "XLK",
        "start"     : "1997-07-01",
        "end"       : "2002-12-31",
        "oos_start" : "2000-03-10",
        "color"     : "#f59e0b",
        "direction" : "SHORT",
    },
}


# ── 데이터 ────────────────────────────────────────────────────────────────

def fetch_returns(ticker, start, end):
    raw = yf.download(ticker, start=start, end=end,
                      auto_adjust=True, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    if ticker in close.columns:
        return close[ticker].pct_change().dropna()
    return close.squeeze().pct_change().dropna()


# ── Layer 5: GJR-GARCH ───────────────────────────────────────────────────

def fit_gjr_garch(returns: pd.Series):
    """
    GJR-GARCH(1,1) 피팅.
    음의 충격이 양의 충격보다 변동성을 더 높이는 비대칭성을 잡음.
    반환: (표준화잔차 z_t, 조건부변동성 sigma_t, 모델 결과)
    """
    r_pct = returns * 100  # arch는 퍼센트 수익률 선호
    model = arch_model(r_pct, vol="GARCH", p=1, o=1, q=1, dist="studentst")
    res   = model.fit(disp="off", show_warning=False)
    sigma_t = res.conditional_volatility / 100      # 원래 스케일로
    z_t     = (returns.values - res.params["mu"] / 100) / sigma_t.values
    z_series = pd.Series(z_t, index=returns.index, name="z_t")
    s_series = pd.Series(sigma_t.values, index=returns.index, name="sigma_t")
    return z_series, s_series, res


# ── 손실 시계열 ────────────────────────────────────────────────────────────

def get_losses(z_t: pd.Series) -> pd.Series:
    """표준화 잔차에서 손실(음의 수익) 추출: L_t = -z_t, L_t > 0 만"""
    return (-z_t).rename("loss")


# ── 진단 도구 1: MEF plot ─────────────────────────────────────────────────

def compute_mef(losses: pd.Series, n_points: int = 80):
    """
    Mean Excess Function: e(u) = E[X - u | X > u]

    GPD 데이터에서 e(u)는 u에 대해 선형.
    선형 구간이 시작되는 u = u* (임계값 후보).
    """
    L = np.sort(losses.dropna().values)
    u_min = np.percentile(L, 50)
    u_max = np.percentile(L, 97)
    u_grid = np.linspace(u_min, u_max, n_points)

    mef_vals, n_exceed = [], []
    for u in u_grid:
        exc = L[L > u] - u
        if len(exc) >= 5:
            mef_vals.append(float(exc.mean()))
            n_exceed.append(len(exc))
        else:
            mef_vals.append(np.nan)
            n_exceed.append(0)

    return pd.DataFrame({
        "u"      : u_grid,
        "mef"    : mef_vals,
        "n_exceed": n_exceed,
    })


# ── 진단 도구 2: Hill plot ────────────────────────────────────────────────

def compute_hill(losses: pd.Series, k_max: int = None):
    """
    Hill Estimator: H_k = (1/k) Σ log(X_(i)/X_(k+1))  for i=1..k

    k에 따라 안정적인 plateau가 나타나는 구간 → 그 H_k = ξ 추정값
    해당 k에 대응하는 X_(k+1) = u* 후보
    """
    L = np.sort(losses.dropna().values)[::-1]  # 내림차순
    n = len(L)
    if k_max is None:
        k_max = min(int(n * 0.4), 300)

    ks, hills, u_vals = [], [], []
    for k in range(5, k_max + 1):
        if L[k] <= 0:
            continue
        h = np.mean(np.log(L[:k] / L[k]))
        ks.append(k)
        hills.append(float(h))
        u_vals.append(float(L[k]))

    return pd.DataFrame({"k": ks, "hill": hills, "u": u_vals})


# ── 진단 도구 3: Parameter Stability ─────────────────────────────────────

def compute_stability(losses: pd.Series, n_points: int = 60):
    """
    여러 u에서 GPD를 피팅해 ξ(u)와 β*(u) = β(u) - ξ(u)·u를 추적.
    두 파라미터가 안정되는 구간의 좌측 경계 = u*
    """
    L = losses.dropna().values
    u_min = np.percentile(L, 55)
    u_max = np.percentile(L, 95)
    u_grid = np.linspace(u_min, u_max, n_points)

    xi_list, beta_list, beta_star_list, n_list = [], [], [], []
    for u in u_grid:
        exc = L[L > u] - u
        if len(exc) < 10:
            xi_list.append(np.nan)
            beta_list.append(np.nan)
            beta_star_list.append(np.nan)
            n_list.append(len(exc))
            continue
        try:
            xi, _, beta = genpareto.fit(exc, floc=0)
            xi_list.append(float(xi))
            beta_list.append(float(beta))
            beta_star_list.append(float(beta - xi * u))
            n_list.append(len(exc))
        except Exception:
            xi_list.append(np.nan)
            beta_list.append(np.nan)
            beta_star_list.append(np.nan)
            n_list.append(len(exc))

    return pd.DataFrame({
        "u"        : u_grid,
        "xi"       : xi_list,
        "beta"     : beta_list,
        "beta_star": beta_star_list,
        "n_exceed" : n_list,
    })


# ── u* 자동 선택 ──────────────────────────────────────────────────────────

def select_threshold(stability_df: pd.DataFrame,
                     hill_df: pd.DataFrame,
                     min_exceed: int = 20) -> float:
    """
    세 진단 도구를 종합해 u* 자동 선택.

    전략:
      1) stability_df에서 xi와 beta_star가 안정적인 구간 찾기
         (롤링 표준편차가 낮고 n_exceed >= min_exceed)
      2) hill_df에서 plateau 구간의 u 값과 비교
      3) 보수적으로 더 높은 u 선택 (꼬리 순수성 우선)
    """
    sdf = stability_df.dropna(subset=["xi", "beta_star"])
    sdf = sdf[sdf["n_exceed"] >= min_exceed].copy()
    if len(sdf) < 5:
        return float(np.percentile(stability_df["u"].dropna(), 80))

    # 롤링 xi 안정성
    sdf["xi_roll_std"] = sdf["xi"].rolling(5, min_periods=3).std()
    stable = sdf[sdf["xi_roll_std"] < sdf["xi_roll_std"].quantile(0.35)]
    if len(stable) == 0:
        stable = sdf

    u_stability = float(stable["u"].iloc[0])

    # Hill plateau: H_k 분산이 낮은 구간
    hdf = hill_df.dropna()
    if len(hdf) > 10:
        hdf["h_roll_std"] = hdf["hill"].rolling(10, min_periods=5).std()
        plateau = hdf[hdf["h_roll_std"] < hdf["h_roll_std"].quantile(0.30)]
        if len(plateau) > 0:
            u_hill = float(plateau["u"].mean())
        else:
            u_hill = u_stability
    else:
        u_hill = u_stability

    # 두 추정의 평균 (보수적 선택)
    u_star = (u_stability + u_hill) / 2
    return round(float(u_star), 6)


# ── GPD 피팅 ──────────────────────────────────────────────────────────────

def fit_gpd(losses: pd.Series, u_star: float):
    """u_star 초과 손실에 GPD 피팅"""
    L = losses.dropna().values
    exc = L[L > u_star] - u_star
    n_total  = len(L)
    n_exceed = len(exc)
    zeta_u   = n_exceed / n_total   # P(L > u*)

    if n_exceed < 10:
        return None

    xi, _, beta = genpareto.fit(exc, floc=0)
    return {
        "xi"      : float(xi),
        "beta"    : float(beta),
        "u_star"  : float(u_star),
        "n_exceed": int(n_exceed),
        "n_total" : int(n_total),
        "zeta_u"  : float(zeta_u),
    }


# ── EVT 기반 VaR / ES ────────────────────────────────────────────────────

def compute_evt_risk(gpd_params: dict, alpha: float = 0.99):
    """
    POT 방법으로 고분위 VaR, ES 계산.

    VaR_p = u + (β/ξ) · [(ζ_u/(1-p))^ξ - 1]
    ES_p  = VaR_p/(1-ξ) + (β - ξu)/(1-ξ)

    alpha = 신뢰수준 (예: 0.99 → 99% VaR/ES)
    """
    xi    = gpd_params["xi"]
    beta  = gpd_params["beta"]
    u     = gpd_params["u_star"]
    zeta  = gpd_params["zeta_u"]

    p = alpha
    if abs(xi) < 1e-8:  # ξ ≈ 0: 지수 꼬리
        var_p = u - beta * np.log((1 - p) / zeta)
        es_p  = var_p + beta
    else:
        var_p = u + (beta / xi) * ((zeta / (1 - p)) ** xi - 1)
        es_p  = var_p / (1 - xi) + (beta - xi * u) / (1 - xi)

    return {
        "alpha"  : alpha,
        "var_evt": float(round(var_p, 6)),
        "es_evt" : float(round(es_p, 6)),
    }


# ── 꼬리 대칭성 분석 (닷컴 vs GFC 구분) ──────────────────────────────────

def analyze_tail_asymmetry(z_t: pd.Series, u_pct: float = 85):
    """
    손실 꼬리(left tail)와 이익 꼬리(right tail) ξ를 따로 추정.

    GFC:   xi_loss >> xi_gain  → 단방향 하락 → SHORT 유효
    닷컴:  xi_loss ≈ xi_gain   → 양방향 → SHORT 청산 위험
    """
    L = z_t.dropna().values
    losses = -L[L < 0]   # 손실: 양수화
    gains  =  L[L > 0]   # 이익: 양수

    u_loss = np.percentile(losses, u_pct)
    u_gain = np.percentile(gains,  u_pct)

    results = {}
    for side, data, u in [("loss", losses, u_loss), ("gain", gains, u_gain)]:
        exc = data[data > u] - u
        if len(exc) >= 10:
            xi, _, beta = genpareto.fit(exc, floc=0)
            results[side] = {
                "xi"     : float(round(xi, 4)),
                "beta"   : float(round(beta, 4)),
                "u"      : float(round(u, 6)),
                "n_exceed": len(exc),
            }
        else:
            results[side] = {"xi": np.nan, "beta": np.nan,
                             "u": u, "n_exceed": len(exc)}

    xi_loss = results["loss"]["xi"]
    xi_gain = results["gain"]["xi"]
    if not np.isnan(xi_loss) and not np.isnan(xi_gain):
        asymmetry = float(round(xi_loss - xi_gain, 4))
        # > 0: 손실 꼬리 더 두꺼움 (SHORT 신호 신뢰)
        # ≈ 0: 양방향 대칭 (SHORT 청산 위험)
    else:
        asymmetry = np.nan

    results["asymmetry"] = asymmetry
    results["interpretation"] = (
        "손실 꼬리 우세 (단방향 하락 -- SHORT/LONG 신뢰)"
        if (not np.isnan(asymmetry) and asymmetry > 0.05)
        else "대칭 꼬리 (양방향 극단 -- 포지션 청산 위험)"
        if (not np.isnan(asymmetry) and asymmetry < -0.05)
        else "중립 (판단 불충분)"
    )
    return results


# ── 시각화 ────────────────────────────────────────────────────────────────

def plot_diagnostics(name, cfg, mef_df, hill_df, stab_df,
                     u_star, gpd_params, asym, losses, z_t):
    fig = plt.figure(figsize=(20, 18))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.38)
    color = cfg["color"]

    # ── 1. MEF plot ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    valid = mef_df.dropna(subset=["mef"])
    ax1.plot(valid["u"], valid["mef"], color=color, linewidth=1.6)
    ax1.axvline(u_star, color="white", linestyle="--", linewidth=1.2,
                label=f"u* = {u_star:.4f}")
    ax1.set_title("MEF Plot — 선형 구간 시작 = u*\n"
                  "e(u) = E[X-u | X>u] : GPD에서 u에 대해 선형",
                  fontsize=9)
    ax1.set_xlabel("u (threshold)", fontsize=8)
    ax1.set_ylabel("e(u)", fontsize=8)
    ax1.legend(fontsize=8)
    ax1.tick_params(labelsize=8)
    ax1.set_facecolor("#111827")
    ax1.grid(alpha=0.2)

    # ── 2. Hill plot ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    hdf = hill_df.dropna()
    ax2.plot(hdf["k"], hdf["hill"], color=color, linewidth=1.2, alpha=0.9)
    ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    if gpd_params:
        ax2.axhline(gpd_params["xi"], color="white", linewidth=1.0,
                    linestyle=":", label=f"GPD ξ = {gpd_params['xi']:.3f}")
    ax2.set_title("Hill Plot — plateau 구간 H_k = ξ 추정값\n"
                  "안정 구간의 H_k = 꼬리 지수",
                  fontsize=9)
    ax2.set_xlabel("k (상위 k개 사용)", fontsize=8)
    ax2.set_ylabel("H_k (Hill Estimator)", fontsize=8)
    ax2.legend(fontsize=8)
    ax2.tick_params(labelsize=8)
    ax2.set_facecolor("#111827")
    ax2.grid(alpha=0.2)

    # ── 3. Parameter Stability (ξ) ───────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    sdf = stab_df.dropna(subset=["xi"])
    ax3.plot(sdf["u"], sdf["xi"], color=color, linewidth=1.4, label="ξ(u)")
    ax3.axvline(u_star, color="white", linestyle="--", linewidth=1.2,
                label=f"u* = {u_star:.4f}")
    ax3.axhline(0, color="gray", linewidth=0.5)
    ax3.set_title("Parameter Stability — ξ(u)\n"
                  "안정 구간의 좌측 경계 = u*",
                  fontsize=9)
    ax3.set_xlabel("u", fontsize=8)
    ax3.set_ylabel("ξ (tail index)", fontsize=8)
    ax3.legend(fontsize=8)
    ax3.tick_params(labelsize=8)
    ax3.set_facecolor("#111827")
    ax3.grid(alpha=0.2)

    # ── 4. Parameter Stability (β*) ──────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    sdf2 = stab_df.dropna(subset=["beta_star"])
    ax4.plot(sdf2["u"], sdf2["beta_star"], color="#f59e0b",
             linewidth=1.4, label="beta*(u) = beta - xi*u")
    ax4.axvline(u_star, color="white", linestyle="--", linewidth=1.2,
                label=f"u* = {u_star:.4f}")
    ax4.set_title("Parameter Stability — beta*(u)\n"
                  "u-불변 형태: 안정 구간이 동일해야 함",
                  fontsize=9)
    ax4.set_xlabel("u", fontsize=8)
    ax4.set_ylabel("beta* = beta - xi*u", fontsize=8)
    ax4.legend(fontsize=8)
    ax4.tick_params(labelsize=8)
    ax4.set_facecolor("#111827")
    ax4.grid(alpha=0.2)

    # ── 5. 손실 분포: 경험적 vs GPD ─────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    L = losses.dropna().values
    ax5.hist(L, bins=60, density=True, color=color, alpha=0.4,
             label="경험적 손실 분포")
    if gpd_params:
        xi_h  = gpd_params["xi"]
        beta_h = gpd_params["beta"]
        u_h   = gpd_params["u_star"]
        x_tail = np.linspace(u_h, np.percentile(L, 99.5), 100)
        gpd_pdf = genpareto.pdf(x_tail - u_h, c=xi_h, scale=beta_h)
        gpd_pdf_scaled = gpd_pdf * gpd_params["zeta_u"]
        ax5.plot(x_tail, gpd_pdf_scaled, color="white", linewidth=1.8,
                 label=f"GPD tail (ξ={xi_h:.3f}, β={beta_h:.3f})")
        ax5.axvline(u_h, color="white", linestyle="--", linewidth=1,
                    label=f"u* = {u_h:.4f}")
    ax5.set_title("손실 분포: 경험적(본체) + GPD(꼬리)\n"
                  "u* 우측이 semi-parametric 꼬리 영역",
                  fontsize=9)
    ax5.set_xlabel("Loss (표준화 잔차 기준)", fontsize=8)
    ax5.legend(fontsize=8)
    ax5.tick_params(labelsize=8)
    ax5.set_facecolor("#111827")
    ax5.grid(alpha=0.2)

    # ── 6. 꼬리 비대칭성 (손실 vs 이익) ─────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    xi_loss = asym["loss"]["xi"] if not np.isnan(asym["loss"]["xi"]) else 0
    xi_gain = asym["gain"]["xi"] if not np.isnan(asym["gain"]["xi"]) else 0
    bars = ax6.bar(["손실 꼬리\n(xi_loss)", "이익 꼬리\n(xi_gain)"],
                   [xi_loss, xi_gain],
                   color=["#ef4444", "#10b981"], alpha=0.8, width=0.5)
    ax6.axhline(0, color="gray", linewidth=0.5)
    for bar, val in zip(bars, [xi_loss, xi_gain]):
        ax6.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", fontsize=10, color="white")
    asym_val = asym.get("asymmetry", 0) or 0
    ax6.set_title(
        f"꼬리 비대칭성 분석 (asymmetry = {asym_val:+.3f})\n"
        f"{asym.get('interpretation', '')}",
        fontsize=9,
    )
    ax6.set_ylabel("ξ (tail index)", fontsize=8)
    ax6.tick_params(labelsize=8)
    ax6.set_facecolor("#111827")
    ax6.grid(alpha=0.2, axis="y")

    fig.suptitle(
        f"EVT Bridge Diagnostics — {cfg['label']}\n"
        f"u* = {u_star:.4f}  |  "
        + (f"GPD ξ = {gpd_params['xi']:.3f}  β = {gpd_params['beta']:.3f}"
           if gpd_params else "GPD 피팅 불가"),
        fontsize=11, weight="bold", color="white",
    )
    fig.patch.set_facecolor("#030712")
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.spines["bottom"].set_color("#374151")
        ax.spines["left"].set_color("#374151")
        ax.spines["top"].set_color("#374151")
        ax.spines["right"].set_color("#374151")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    out = FIGURES_DIR / f"evt_bridge_{name}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="#030712")
    plt.close()
    print(f"  저장: {out}")


# ── JSON 내보내기 ─────────────────────────────────────────────────────────

def export_json(all_results: dict):
    import json

    payload = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "description": "EVT Bridge — Layer 3 Semi-Parametric Threshold",
        "crises": {},
    }

    for name, res in all_results.items():
        gpd  = res.get("gpd")
        asym = res.get("asymmetry", {})
        risk = res.get("risk_99", {})
        risk95 = res.get("risk_95", {})

        payload["crises"][name] = {
            "label"    : res["label"],
            "etf"      : res["etf"],
            "color"    : res["color"],
            "direction": res["direction"],
            "garch_params": res.get("garch_params", {}),
            "threshold": {
                "u_star"         : round(res["u_star"], 6),
                "u_star_quantile": round(res["u_star_quantile"], 1),
                "n_exceed"       : int(gpd["n_exceed"]) if gpd else None,
                "n_total"        : int(gpd["n_total"]) if gpd else None,
                "zeta_u"         : round(gpd["zeta_u"], 4) if gpd else None,
            },
            "gpd": {
                "xi"  : round(gpd["xi"], 4) if gpd else None,
                "beta": round(gpd["beta"], 4) if gpd else None,
                "interpretation": (
                    "두꺼운 꼬리 (Pareto형)" if gpd and gpd["xi"] > 0.3
                    else "중간 꼬리" if gpd and gpd["xi"] > 0.1
                    else "얇은 꼬리 (지수형에 가까움)"
                ) if gpd else None,
            },
            "tail_asymmetry": {
                "xi_loss"       : round(float(asym.get("loss", {}).get("xi") or 0), 4),
                "xi_gain"       : round(float(asym.get("gain", {}).get("xi") or 0), 4),
                "asymmetry"     : round(float(asym.get("asymmetry") or 0), 4),
                "interpretation": asym.get("interpretation", ""),
            },
            "risk_measures": {
                "var_99" : round(risk.get("var_evt", 0), 5),
                "es_99"  : round(risk.get("es_evt", 0), 5),
                "var_95" : round(risk95.get("var_evt", 0), 5),
                "es_95"  : round(risk95.get("es_evt", 0), 5),
            },
            "diagnostics": {
                "mef": [
                    {"u": round(float(r["u"]), 6),
                     "mef": round(float(r["mef"]), 6),
                     "n": int(r["n_exceed"])}
                    for _, r in res["mef_df"].dropna(subset=["mef"]).iterrows()
                ],
                "hill": [
                    {"k": int(r["k"]),
                     "hill": round(float(r["hill"]), 6),
                     "u": round(float(r["u"]), 6)}
                    for _, r in res["hill_df"].dropna().iterrows()
                    if not np.isnan(r["hill"])
                ],
                "stability": [
                    {"u": round(float(r["u"]), 6),
                     "xi": round(float(r["xi"]), 6) if not np.isnan(r["xi"]) else None,
                     "beta_star": round(float(r["beta_star"]), 6) if not np.isnan(r["beta_star"]) else None,
                     "n": int(r["n_exceed"])}
                    for _, r in res["stab_df"].iterrows()
                ],
            },
        }

    out = OUTPUT_DIR / "evt_bridge.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n  JSON 저장: {out}")
    return payload


# ── 메인 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 68)
    print("  EVT Bridge -- Layer 3 Semi-Parametric Threshold")
    print("=" * 68)

    all_results = {}

    for name, cfg in CRISES.items():
        print(f"\n[{cfg['label']}]")

        # 데이터
        print("  데이터 로드...")
        ret = fetch_returns(cfg["etf"], cfg["start"], cfg["end"])
        if ret is None or len(ret) < 100:
            print("  SKIP: 데이터 부족")
            continue

        # GJR-GARCH
        print("  GJR-GARCH 피팅...")
        try:
            z_t, sigma_t, garch_res = fit_gjr_garch(ret)
            garch_params = {
                "omega": round(float(garch_res.params.get("omega", 0)), 8),
                "alpha": round(float(garch_res.params.get("alpha[1]", 0)), 4),
                "gamma": round(float(garch_res.params.get("gamma[1]", 0)), 4),
                "beta" : round(float(garch_res.params.get("beta[1]", 0)), 4),
                "nu"   : round(float(garch_res.params.get("nu", 0)), 2),
            }
            print(f"  GARCH: alpha={garch_params['alpha']:.3f} "
                  f"gamma={garch_params['gamma']:.3f} "
                  f"beta={garch_params['beta']:.3f}")
        except Exception as e:
            print(f"  GARCH 실패: {e}, EWMA 대체 사용")
            ewma_vol = ret.ewm(span=32).std()
            z_t = ret / ewma_vol.clip(lower=1e-6)
            z_t = z_t.dropna()
            garch_params = {"fallback": "EWMA(span=32)"}

        losses = get_losses(z_t)

        # 진단 도구
        print("  진단 계산 중...")
        mef_df  = compute_mef(losses)
        hill_df = compute_hill(losses)
        stab_df = compute_stability(losses)

        # u* 선택
        u_star = select_threshold(stab_df, hill_df)
        L = losses.dropna().values
        u_pct = float(np.mean(L <= u_star) * 100)
        print(f"  u* = {u_star:.4f}  (P{u_pct:.0f})")

        # GPD 피팅
        gpd_params = fit_gpd(losses, u_star)
        if gpd_params:
            print(f"  GPD: xi={gpd_params['xi']:.3f}  "
                  f"beta={gpd_params['beta']:.3f}  "
                  f"n_exceed={gpd_params['n_exceed']}")
        else:
            print("  GPD 피팅 실패 (초과값 부족)")

        # EVT 기반 리스크
        risk_99, risk_95 = {}, {}
        if gpd_params:
            risk_99 = compute_evt_risk(gpd_params, alpha=0.99)
            risk_95 = compute_evt_risk(gpd_params, alpha=0.95)
            print(f"  ES@99%={risk_99['es_evt']:.4f}  "
                  f"VaR@99%={risk_99['var_evt']:.4f}")

        # 꼬리 비대칭성
        asym = analyze_tail_asymmetry(z_t)
        print(f"  꼬리: loss_xi={asym['loss']['xi']:.3f}  "
              f"gain_xi={asym['gain']['xi']:.3f}  "
              f"-> {asym['interpretation']}")

        # 시각화
        print("  시각화...")
        plot_diagnostics(name, cfg, mef_df, hill_df, stab_df,
                         u_star, gpd_params, asym, losses, z_t)

        all_results[name] = {
            "label"     : cfg["label"],
            "etf"       : cfg["etf"],
            "color"     : cfg["color"],
            "direction" : cfg["direction"],
            "garch_params": garch_params,
            "u_star"    : u_star,
            "u_star_quantile": u_pct,
            "gpd"       : gpd_params,
            "risk_99"   : risk_99,
            "risk_95"   : risk_95,
            "asymmetry" : asym,
            "mef_df"    : mef_df,
            "hill_df"   : hill_df,
            "stab_df"   : stab_df,
        }

    # 요약 테이블
    print("\n" + "=" * 68)
    print(f"  {'위기':<32} {'u*':>7} {'P%':>5} {'xi':>7} {'ES@99':>8} {'비대칭':>8}")
    print("-" * 68)
    for name, res in all_results.items():
        gpd  = res["gpd"]
        asym = res["asymmetry"].get("asymmetry", float("nan"))
        es   = res["risk_99"].get("es_evt", float("nan"))
        print(f"  {res['label']:<32} "
              f"{res['u_star']:>7.4f} "
              f"P{res['u_star_quantile']:>3.0f} "
              f"{gpd['xi']:>7.3f}  " if gpd else "  N/A",
              end="")
        print(f"{es:>8.4f}  {asym:>+8.3f}" if not np.isnan(es) else "  N/A")
    print("=" * 68)

    # JSON 내보내기
    print("\n  JSON 내보내기...")
    export_json(all_results)
    print("\n완료.")

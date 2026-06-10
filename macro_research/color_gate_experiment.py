"""
color_gate_experiment.py
Research: A+B 동시 적용 실험 — color(VIX) 4차 계수를 게이트 타이밍에 적용.
  Method A : IS 기간 color → GATE_THRESH 정적 조정
  Method B : rolling color → gate_score에 color_comp 추가
  A+B      : 둘 다 동시 적용 (캘리브레이션 없음)

Usage:
  python color_gate_experiment.py [TICKER] [START] [OOS_START] [DIRECTION]
  python color_gate_experiment.py XLK 2019-01-01 2019-07-01 SHORT
"""

import warnings; warnings.filterwarnings("ignore")
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import date

# ── CLI args ──────────────────────────────────────────────────────────────────
TICKER    = sys.argv[1] if len(sys.argv) > 1 else "XLK"
START     = sys.argv[2] if len(sys.argv) > 2 else "2019-01-01"
OOS_START = sys.argv[3] if len(sys.argv) > 3 else "2019-07-01"
DIRECTION = sys.argv[4] if len(sys.argv) > 4 else "SHORT"
END       = date.today().isoformat()

# ── Gate parameters (quantify.py와 동일) ─────────────────────────────────────
EWMA_SPAN   = 32
RV_WIN      = 20
BETA_WIN    = 30
GATE_WINDOW = 15
GATE_CONSEC = 5
GATE_THRESH = 0.45

# ── Color parameters ──────────────────────────────────────────────────────────
COLOR_WINDOW = 63       # rolling 회귀 lookback
COLOR_WARN   = -0.001   # IS color 이 값 아래면 Method A 발동
COLOR_PENALTY = 0.05    # Method A 최대 threshold 상승폭
COLOR_WEIGHT  = 0.15    # Method B color_comp 가중치


# ── Data ──────────────────────────────────────────────────────────────────────

def fetch_data(ticker, start, end):
    syms = [ticker, "SPY", "^VIX"]
    raw  = yf.download(syms, start=start, end=end,
                       auto_adjust=True, progress=False, timeout=30)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    out = {}
    for t in syms:
        if t in close.columns:
            s = close[t].dropna()
            if len(s) > 20:
                out[t] = s
    return out


# ── 4th-order polynomial regression helpers ───────────────────────────────────

def poly4_color(y: np.ndarray, x: np.ndarray) -> float:
    """y ~ [1, x, x², x³, x⁴] → return color (4차 계수)."""
    X = np.column_stack([np.ones(len(y)), x, x**2, x**3, x**4])
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return float(coeffs[4])


def compute_is_color(sec_ret: pd.Series, vix: pd.Series, is_mask) -> float:
    """IS 기간 전체를 한 번 회귀 → IS color 단일값."""
    vix_chg = vix.pct_change().dropna()
    idx     = sec_ret[is_mask].index.intersection(vix_chg.index)
    if len(idx) < 30:
        return np.nan
    try:
        return poly4_color(sec_ret.loc[idx].values, vix_chg.loc[idx].values)
    except Exception:
        return np.nan


def compute_rolling_color(sec_ret: pd.Series, vix: pd.Series,
                          window: int = COLOR_WINDOW) -> pd.Series:
    """Rolling window 4차 회귀 → color 시계열."""
    vix_chg = vix.pct_change().dropna()
    idx     = sec_ret.index.intersection(vix_chg.index)
    s = sec_ret.loc[idx].values
    v = vix_chg.loc[idx].values
    n = len(s)
    colors = np.full(n, np.nan)

    for i in range(window, n):
        try:
            colors[i] = poly4_color(s[i - window:i], v[i - window:i])
        except Exception:
            pass

    return pd.Series(colors, index=idx, name="color")


# ── Gate score building ───────────────────────────────────────────────────────

def build_base_components(s, m, vix, idx2, direction):
    """beta_comp, vrp_comp, dir_comp — quantify.py와 동일 로직."""
    beta = (s.rolling(BETA_WIN).cov(m) / m.rolling(BETA_WIN).var()).dropna()

    b63     = (s.rolling(63).cov(m) / m.rolling(63).var()).dropna().reindex(idx2).ffill()
    res     = s - b63 * m
    idio    = res.ewm(span=EWMA_SPAN, min_periods=20).std() * np.sqrt(252)
    vix_a   = vix.reindex(idx2).ffill() / 100.0
    capm_iv = np.sqrt((b63 * vix_a) ** 2 + idio ** 2)
    rv_ann  = s.rolling(RV_WIN).std() * np.sqrt(252)
    vrp     = (capm_iv - rv_ann).dropna()

    idx3 = beta.index.intersection(vrp.index).intersection(idx2)
    b    = beta.reindex(idx3)
    v    = vrp.reindex(idx3)
    excs = s.reindex(idx3) - m.reindex(idx3)

    if direction == "LONG":
        beta_comp = (b < -0.05).astype(float).rolling(GATE_WINDOW, min_periods=5).mean()
        dir_comp  = (excs > 0).astype(float).rolling(GATE_WINDOW, min_periods=5).mean()
    else:
        beta_comp = (b > 1.50).astype(float).rolling(GATE_WINDOW, min_periods=5).mean()
        dir_comp  = (excs < 0).astype(float).rolling(GATE_WINDOW, min_periods=5).mean()

    vrp_comp = (v < 0).astype(float).rolling(GATE_WINDOW, min_periods=5).mean()
    return beta_comp, vrp_comp, dir_comp, idx3


def gate_open_series(score: pd.Series, thresh: float) -> pd.Series:
    above = (score >= thresh).astype(float)
    return above.rolling(GATE_CONSEC, min_periods=GATE_CONSEC).min().fillna(0).astype(bool)


# ── Main experiment ───────────────────────────────────────────────────────────

def run_experiment(data, ticker, direction, oos_start):
    sec_ret = data[ticker].pct_change().dropna()
    spy_ret = data["SPY"].pct_change().dropna()
    vix     = data["^VIX"]

    idx  = sec_ret.index.intersection(spy_ret.index)
    s, m = sec_ret.loc[idx], spy_ret.loc[idx]
    oos_dt = pd.Timestamp(oos_start)

    # Base components
    beta_comp, vrp_comp, dir_comp, idx3 = build_base_components(s, m, vix, idx, direction)

    # ── Method A: IS color → threshold adjustment ─────────────────────────────
    is_color = compute_is_color(sec_ret, vix, sec_ret.index < oos_dt)
    color_adj = 0.0
    if np.isfinite(is_color) and is_color < COLOR_WARN:
        # 비율 조정 (캘리브레이션 없이 그대로)
        color_adj = COLOR_PENALTY * min(3.0, abs(is_color / COLOR_WARN))
    gate_thresh_a = GATE_THRESH + color_adj

    # ── Method B: rolling color → color_comp ──────────────────────────────────
    print("  Rolling 4th-order regression 계산 중 (~10초)...")
    color_series = compute_rolling_color(s.reindex(idx3), vix.reindex(idx3).ffill())
    color_comp   = (color_series > 0).astype(float).rolling(GATE_WINDOW, min_periods=5).mean()
    color_comp   = color_comp.reindex(idx3)

    # Gate scores
    score_base = 0.40 * beta_comp + 0.35 * vrp_comp + 0.25 * dir_comp

    # B: 기존 weights를 (1-COLOR_WEIGHT) 비율로 줄이고 color_comp 추가
    w = 1.0 - COLOR_WEIGHT
    score_b = (0.40 * w * beta_comp
             + 0.35 * w * vrp_comp
             + 0.25 * w * dir_comp
             + COLOR_WEIGHT * color_comp)

    variants = {
        "baseline":  {"score": score_base, "thresh": GATE_THRESH,   "label": "Baseline"},
        "method_a":  {"score": score_base, "thresh": gate_thresh_a, "label": "Method A"},
        "method_b":  {"score": score_b,    "thresh": GATE_THRESH,   "label": "Method B"},
        "method_ab": {"score": score_b,    "thresh": gate_thresh_a, "label": "A + B"},
    }
    for v in variants.values():
        v["gate_open"] = gate_open_series(v["score"], v["thresh"])

    return dict(
        variants     = variants,
        idx3         = idx3,
        oos_dt       = oos_dt,
        is_color     = is_color,
        color_adj    = color_adj,
        gate_thresh_a= gate_thresh_a,
        color_series = color_series,
        color_comp   = color_comp,
        beta_comp    = beta_comp,
        vrp_comp     = vrp_comp,
        dir_comp     = dir_comp,
        sec_ret      = s.reindex(idx3),
    )


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(res, ticker, direction):
    oos_dt   = res["oos_dt"]
    sec_ret  = res["sec_ret"]
    oos_ret  = sec_ret[sec_ret.index >= oos_dt]
    is_color = res["is_color"]

    print(f"\n{'='*72}")
    print(f"  COLOR GATE EXPERIMENT  |  {ticker}  |  {direction}")
    print(f"  IS color(VIX) = {is_color:.6f}   COLOR_WARN = {COLOR_WARN}")
    print(f"  Threshold adj = +{res['color_adj']:.4f}  →  A/AB thresh = {res['gate_thresh_a']:.4f}")
    print(f"{'='*72}")
    print(f"  {'Variant':<12} {'Thresh':>7} {'Open%':>8} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8} {'N days':>7}")
    print(f"  {'-'*64}")

    for vname, v in res["variants"].items():
        go    = v["gate_open"].reindex(oos_ret.index, fill_value=False)
        gated = oos_ret[go]
        opct  = go.mean() * 100

        if len(gated) > 1:
            cum  = float((1 + gated).prod())
            cagr = float(cum ** (252 / len(gated)) - 1)
            sh   = float(gated.mean() / gated.std() * 252 ** 0.5) if gated.std() > 0 else 0
            cp   = (1 + gated).cumprod()
            mdd  = float((cp / cp.cummax() - 1).min())
            nd   = len(gated)
        else:
            cagr = sh = mdd = 0; nd = 0

        flag = " ← SUPPRESSED" if nd == 0 else ""
        print(f"  {v['label']:<12} {v['thresh']:>7.3f} {opct:>7.1f}% {cagr:>8.2%} {sh:>8.2f} {mdd:>8.2%} {nd:>7}{flag}")

    # OOS 전체 buy-and-hold 비교
    if len(oos_ret) > 1:
        cum  = float((1 + oos_ret).prod())
        cagr = float(cum ** (252 / len(oos_ret)) - 1)
        sh   = float(oos_ret.mean() / oos_ret.std() * 252**0.5) if oos_ret.std() > 0 else 0
        cp   = (1 + oos_ret).cumprod()
        mdd  = float((cp / cp.cummax() - 1).min())
        print(f"  {'[B&H full]':<12} {'--':>7} {'100.0':>7}% {cagr:>8.2%} {sh:>8.2f} {mdd:>8.2%} {len(oos_ret):>7}")
    print(f"{'='*72}\n")


# ── Plot ──────────────────────────────────────────────────────────────────────

def save_plot(res, ticker, direction):
    oos_dt  = res["oos_dt"]
    sec_ret = res["sec_ret"]
    oos_ret = sec_ret[sec_ret.index >= oos_dt]
    variants = res["variants"]

    DARK  = "#0f0f1a"
    PANEL = "#1a1a2e"
    CLR   = {"baseline": "#4A90D9", "method_a": "#F5A623",
              "method_b": "#7ED321", "method_ab": "#D0021B"}

    fig, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=True)
    fig.patch.set_facecolor(DARK)
    fig.suptitle(
        f"Color Gate Experiment — {ticker} ({direction})\n"
        f"IS color(VIX)={res['is_color']:.6f}  |  "
        f"color_adj=+{res['color_adj']:.4f}  |  A/AB thresh={res['gate_thresh_a']:.4f}  |  "
        f"OOS start: {res['oos_dt'].date()}",
        fontsize=10, color="white"
    )

    # ── Row 0: gate scores ────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(variants["baseline"]["score"], color=CLR["baseline"], lw=1, alpha=0.8, label="Baseline score")
    ax.plot(variants["method_b"]["score"],  color=CLR["method_b"],  lw=1, alpha=0.8, label="B/AB score (color_comp)")
    ax.axhline(GATE_THRESH,           color="gray",       ls="--", lw=0.8, label=f"Base thresh {GATE_THRESH:.2f}")
    ax.axhline(res["gate_thresh_a"],  color=CLR["method_a"], ls="--", lw=0.8, label=f"A thresh {res['gate_thresh_a']:.3f}")
    ax.axvline(oos_dt, color="white", ls=":", lw=0.8)
    ax.set_ylabel("Gate Score", color="white", fontsize=8)
    ax.legend(fontsize=7, loc="upper left")

    # ── Row 1: rolling color coefficient ─────────────────────────────────────
    ax = axes[1]
    ax.plot(res["color_series"], color="#B8B8FF", lw=1, alpha=0.85, label="color coeff (rolling 63d)")
    ax.fill_between(res["color_series"].index, 0, res["color_series"],
                    where=res["color_series"] < 0, color="#D0021B", alpha=0.25, label="color < 0 (gamma fading)")
    ax.axhline(0,           color="white",       ls="--", lw=0.7)
    ax.axhline(COLOR_WARN,  color=CLR["method_a"], ls=":",  lw=0.8, label=f"warn {COLOR_WARN}")
    ax.axvline(oos_dt, color="white", ls=":", lw=0.8)
    ax.set_ylabel("color(VIX)", color="white", fontsize=8)
    ax.legend(fontsize=7, loc="upper left")

    # ── Row 2: gate open timeline ─────────────────────────────────────────────
    ax = axes[2]
    offsets = {"baseline": 0.0, "method_a": 0.22, "method_b": 0.44, "method_ab": 0.66}
    for vname, v in variants.items():
        off = offsets[vname]
        ax.fill_between(v["gate_open"].index, off, off + 0.18,
                        where=v["gate_open"],
                        color=CLR[vname], alpha=0.75, label=v["label"])
    ax.axvline(oos_dt, color="white", ls=":", lw=0.8)
    ax.set_yticks([0.09, 0.31, 0.53, 0.75])
    ax.set_yticklabels(["Baseline", "A", "B", "A+B"], color="white", fontsize=7)
    ax.set_ylabel("Gate Open", color="white", fontsize=8)
    ax.legend(fontsize=7, loc="upper left", ncol=2)

    # ── Row 3: OOS cumulative returns ─────────────────────────────────────────
    ax = axes[3]
    for vname, v in variants.items():
        go    = v["gate_open"].reindex(oos_ret.index, fill_value=False)
        gated = oos_ret[go]
        if len(gated) > 1:
            cum = (1 + gated).cumprod() - 1
            ax.plot(cum, color=CLR[vname], lw=1.3, label=f"{v['label']} ({cum.iloc[-1]:+.1%})")
    ax.axhline(0, color="white", ls="--", lw=0.5)
    ax.set_ylabel("Cum. Return (OOS gate-open)", color="white", fontsize=8)
    ax.legend(fontsize=7, loc="upper left")

    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors="white", labelsize=7)
        ax.grid(True, alpha=0.12, color="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    plt.tight_layout()
    outpath = f"color_gate_experiment_{ticker}.png"
    plt.savefig(outpath, dpi=130, bbox_inches="tight", facecolor=DARK)
    print(f"Plot saved → {outpath}")
    return outpath


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nFetching {TICKER} | {START} → {END} | OOS: {OOS_START} | Dir: {DIRECTION}")
    data = fetch_data(TICKER, START, END)
    if TICKER not in data:
        print(f"ERROR: {TICKER} 데이터를 가져오지 못했습니다.")
        sys.exit(1)

    res = run_experiment(data, TICKER, DIRECTION, OOS_START)
    print_report(res, TICKER, DIRECTION)
    save_plot(res, TICKER, DIRECTION)

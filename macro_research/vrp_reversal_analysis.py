"""
vrp_reversal_analysis.py - VRP 반전 원인 추적

질문: 왜 2026년에 VRP 전략이 반전됐는가?

수사 순서:
  1. 반전 시점 — VRP가 언제 음수로 전환됐나 (섹터별)
  2. 반전 섹터의 공통점 — 어떤 섹터가 뒤집혔고 공통 특성은?
  3. RV spike vs IV collapse — 실현변동성이 올랐나, 내재변동성이 내렸나?
  4. 매크로 트리거 — VIX, DXY, 금리 중 어느 변수가 반전과 연결되나?
  5. 구조적 변화 vs 일시적 충격 — 베타 자체가 변했나?
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

from config import OUTPUT_DIR, FIGURES_DIR, OOS_START, SECTOR_ETFS

OOS_DT  = pd.Timestamp(OOS_START)
SECTORS = list(SECTOR_ETFS.keys())


def load():
    returns   = pd.read_parquet(OUTPUT_DIR / "sector_returns.parquet")
    capm_iv   = pd.read_parquet(OUTPUT_DIR / "capm_iv.parquet")
    rv        = pd.read_parquet(OUTPUT_DIR / "realized_vol.parquet")
    macro_lvl = pd.read_parquet(OUTPUT_DIR / "macro_levels.parquet")
    macro_chg = pd.read_parquet(OUTPUT_DIR / "macro_changes.parquet")
    vrp       = pd.read_csv(OUTPUT_DIR / "vrp_capm_series.csv",
                            index_col=0, parse_dates=True)
    iv_spread = pd.read_csv(OUTPUT_DIR / "iv_spread_series.csv",
                            index_col=0, parse_dates=True)

    # VRP 재구성 (capm_iv - rv, aligned)
    vrp_full = capm_iv.subtract(rv.reindex(capm_iv.index), fill_value=np.nan)
    vrp_full = vrp_full[[c for c in SECTORS if c in vrp_full.columns]]

    return returns, capm_iv, rv, macro_lvl, macro_chg, vrp_full, iv_spread


# ---------------------------------------------------------------------------
# 1. VRP 전환 시점 탐지
# ---------------------------------------------------------------------------

def find_reversal_dates(vrp_full: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    각 섹터에서 VRP 10일 이동평균이 0 아래로 처음 꺾인 날짜.
    """
    rows = []
    for sec in SECTORS:
        if sec not in vrp_full.columns:
            continue
        s = vrp_full[sec].dropna()
        ma = s.rolling(window).mean()
        # OOS 구간에서 음수 전환
        oos_ma = ma[ma.index >= OOS_DT]
        neg = oos_ma[oos_ma < 0]
        if len(neg) > 0:
            first_neg = neg.index[0]
            # IS에서의 마지막 양수
            is_ma = ma[ma.index < OOS_DT]
            last_pos_is = is_ma[is_ma > 0]
            is_level = last_pos_is.iloc[-1] if len(last_pos_is) > 0 else np.nan
            rows.append({
                "sector":        sec,
                "reversal_date": first_neg,
                "vrp_at_reversal": vrp_full.loc[first_neg, sec] if first_neg in vrp_full.index else np.nan,
                "vrp_is_mean":   vrp_full.loc[vrp_full.index < OOS_DT, sec].mean(),
                "vrp_oos_mean":  vrp_full.loc[vrp_full.index >= OOS_DT, sec].mean(),
            })
        else:
            rows.append({
                "sector": sec,
                "reversal_date": None,
                "vrp_at_reversal": np.nan,
                "vrp_is_mean":   vrp_full.loc[vrp_full.index < OOS_DT, sec].mean(),
                "vrp_oos_mean":  vrp_full.loc[vrp_full.index >= OOS_DT, sec].mean(),
            })
    return pd.DataFrame(rows).set_index("sector")


# ---------------------------------------------------------------------------
# 2. RV spike vs IV collapse 분해
# ---------------------------------------------------------------------------

def rv_iv_decompose(capm_iv: pd.DataFrame, rv: pd.DataFrame) -> pd.DataFrame:
    """
    OOS 구간에서 VRP 변화량을 delta_IV와 delta_RV로 분해.
    VRP_change = (IV_oos - IV_is) - (RV_oos - RV_is)
               = delta_IV - delta_RV
    delta_IV < 0 → IV 하락이 주범
    delta_RV > 0 → RV 상승이 주범
    """
    rows = []
    for sec in SECTORS:
        if sec not in capm_iv.columns:
            continue
        iv_is  = capm_iv.loc[capm_iv.index <  OOS_DT, sec].mean()
        iv_oos = capm_iv.loc[capm_iv.index >= OOS_DT, sec].mean()
        rv_s   = rv[sec].reindex(capm_iv.index) if sec in rv.columns else pd.Series(dtype=float)
        rv_is  = rv_s[rv_s.index <  OOS_DT].mean()
        rv_oos = rv_s[rv_s.index >= OOS_DT].mean()

        delta_iv  = iv_oos  - iv_is
        delta_rv  = rv_oos  - rv_is
        vrp_delta = delta_iv - delta_rv

        # 주범 판단: |delta_IV| vs |delta_RV|
        if abs(delta_rv) > abs(delta_iv):
            culprit = "RV_spike"
        elif delta_iv < 0:
            culprit = "IV_collapse"
        else:
            culprit = "VRP_expanded"

        rows.append({
            "sector":      sec,
            "delta_IV":    delta_iv,
            "delta_RV":    delta_rv,
            "vrp_delta":   vrp_delta,
            "culprit":     culprit,
            "iv_is":       iv_is,
            "iv_oos":      iv_oos,
            "rv_is":       rv_is,
            "rv_oos":      rv_oos,
        })
    return pd.DataFrame(rows).set_index("sector")


# ---------------------------------------------------------------------------
# 3. 매크로 트리거 분석
# ---------------------------------------------------------------------------

def macro_trigger_analysis(vrp_full: pd.DataFrame,
                            macro_lvl: pd.DataFrame,
                            macro_chg: pd.DataFrame) -> pd.DataFrame:
    """
    OOS 구간에서 각 매크로 변수와 VRP 변화량의 상관관계.
    - 섹터별로 VRP와 VIX level/change, DXY, US10Y의 상관계수
    """
    macro_vars = ["VIX", "US10Y", "US2Y", "DXY"]
    rows = []

    for sec in SECTORS:
        if sec not in vrp_full.columns:
            continue
        vrp_s = vrp_full[sec].dropna()
        vrp_oos = vrp_s[vrp_s.index >= OOS_DT]

        for mv in macro_vars:
            if mv not in macro_lvl.columns:
                continue
            # VRP 변화율 vs 매크로 레벨/변화
            macro_lv = macro_lvl[mv].reindex(vrp_oos.index).ffill()
            macro_ch = macro_chg[mv].reindex(vrp_oos.index) if mv in macro_chg.columns else None

            if len(macro_lv.dropna()) > 10:
                corr_lv = vrp_oos.corr(macro_lv)
                rows.append({
                    "sector": sec, "macro": mv,
                    "type": "level",
                    "corr": corr_lv,
                })
            if macro_ch is not None and len(macro_ch.dropna()) > 10:
                corr_ch = vrp_oos.corr(macro_ch.reindex(vrp_oos.index))
                rows.append({
                    "sector": sec, "macro": mv,
                    "type": "change",
                    "corr": corr_ch,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. 베타 구조 변화
# ---------------------------------------------------------------------------

def beta_shift_analysis(returns: pd.DataFrame) -> pd.DataFrame:
    """
    IS vs OOS 구간에서 각 섹터의 SPY 베타 비교.
    베타가 변하면 CAPM IV 분해 자체가 달라짐.
    """
    spy = returns["SPY"].dropna()
    rows = []
    for sec in SECTORS:
        if sec not in returns.columns:
            continue
        s = returns[sec].dropna()
        common = s.index.intersection(spy.index)
        s, m = s.loc[common], spy.loc[common]

        def _beta(mask):
            sm, mm = s[mask], m[mask]
            if len(sm) < 20:
                return np.nan
            cov = np.cov(sm.values, mm.values)
            return cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else np.nan

        is_mask  = common < OOS_DT
        oos_mask = common >= OOS_DT
        b_is  = _beta(is_mask)
        b_oos = _beta(oos_mask)
        rows.append({
            "sector": sec,
            "beta_is":  b_is,
            "beta_oos": b_oos,
            "beta_shift": b_oos - b_is if not (np.isnan(b_is) or np.isnan(b_oos)) else np.nan,
        })
    return pd.DataFrame(rows).set_index("sector")


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------

def plot_reversal(vrp_full, decomp_df, beta_df, macro_lvl, corr_df):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(20, 20))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)

    # ── (0,0) VRP 시계열 — 섹터별 ────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    colors = plt.cm.tab20(np.linspace(0, 1, len(SECTORS)))
    for sec, c in zip(SECTORS, colors):
        if sec in vrp_full.columns:
            vrp_full[sec].rolling(10).mean().plot(
                ax=ax, label=sec, color=c, linewidth=0.9, alpha=0.85)
    ax.axvline(OOS_DT, color="red", linewidth=1.5, linestyle=":", label="OOS start")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.fill_betweenx([ax.get_ylim()[0] if ax.get_ylim()[0] < -0.1 else -0.1,
                      ax.get_ylim()[1] if ax.get_ylim()[1] > 0.15 else 0.15],
                     OOS_DT, vrp_full.index[-1],
                     alpha=0.06, color="red")
    ax.set_title("VRP 시계열 (10D MA)\n0 이하 = RV > IV (변동성 폭발 구간)", fontsize=9)
    ax.legend(fontsize=5.5, ncol=3)
    ax.set_ylabel("VRP (연환산)", fontsize=8)
    ax.tick_params(labelsize=7)

    # ── (0,1) delta_IV vs delta_RV 막대 ──────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(len(SECTORS))
    secs_sorted = decomp_df.sort_values("vrp_delta").index.tolist()
    dIV = decomp_df["delta_IV"].reindex(secs_sorted).values
    dRV = decomp_df["delta_RV"].reindex(secs_sorted).values
    ax.bar(x - 0.2, dIV, 0.38, label="delta IV (OOS-IS)", color="#1565C0", alpha=0.8)
    ax.bar(x + 0.2, dRV, 0.38, label="delta RV (OOS-IS)", color="#C62828", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(secs_sorted, rotation=30, fontsize=8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title("IS -> OOS: IV 변화 vs RV 변화\n(양수=상승, 음수=하락)", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_ylabel("Annualized vol change", fontsize=8)

    # ── (1,0) VIX 레벨 + VRP 반전 섹터 오버레이 ─────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax2 = ax.twinx()
    vix = macro_lvl["VIX"].reindex(vrp_full.index).ffill()
    vix.plot(ax=ax, color="gray", linewidth=0.8, alpha=0.6, label="VIX level")
    ax.axvline(OOS_DT, color="red", linewidth=1.2, linestyle=":")
    ax.set_ylabel("VIX", fontsize=8)
    ax.tick_params(labelsize=7)

    # 반전 섹터 (OOS mean VRP < 0)
    reversal_secs = [s for s in SECTORS
                     if s in vrp_full.columns
                     and vrp_full.loc[vrp_full.index >= OOS_DT, s].mean() < 0]
    for sec in reversal_secs:
        vrp_full[sec].rolling(5).mean().plot(ax=ax2, linewidth=1.2, label=f"{sec} VRP",
                                              alpha=0.85)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("VRP (반전 섹터)", fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.legend(fontsize=7, loc="lower right")
    ax.set_title(f"VIX vs VRP 반전 섹터({', '.join(reversal_secs)})\nVIX 급등 시 VRP도 함께 음전환?", fontsize=9)

    # ── (1,1) 베타 변화 ───────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    beta_is  = beta_df["beta_is"].reindex(SECTORS)
    beta_oos = beta_df["beta_oos"].reindex(SECTORS)
    beta_shift = beta_df["beta_shift"].reindex(SECTORS)
    x = np.arange(len(SECTORS))
    ax.bar(x - 0.2, beta_is.values,  0.38,
           label="beta IS (2025)",  color="#1565C0", alpha=0.8)
    ax.bar(x + 0.2, beta_oos.values, 0.38,
           label="beta OOS (2026)", color="#E65100", alpha=0.8)
    for i, (bi, bo) in enumerate(zip(beta_is.values, beta_oos.values)):
        shift = bo - bi
        if not np.isnan(shift):
            ax.annotate(f"{shift:+.2f}", xy=(i + 0.2, bo),
                        ha="center", va="bottom", fontsize=6,
                        color="red" if abs(shift) > 0.15 else "gray")
    ax.set_xticks(x)
    ax.set_xticklabels(SECTORS, rotation=30, fontsize=8)
    ax.axhline(1.0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_title("SPY 베타: IS vs OOS\n빨간 숫자 = 큰 변화 (CAPM IV 분해에 영향)", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_ylabel("Beta to SPY", fontsize=8)

    # ── (2,0) VIX level과 VRP 산점도 (OOS 구간) ──────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    vix_oos = macro_lvl["VIX"].reindex(vrp_full.index).ffill()
    vix_oos = vix_oos[vix_oos.index >= OOS_DT]
    colors2 = plt.cm.tab20(np.linspace(0, 1, len(SECTORS)))
    for sec, c in zip(SECTORS, colors2):
        if sec not in vrp_full.columns:
            continue
        v = vrp_full[sec].reindex(vix_oos.index).dropna()
        common = v.index.intersection(vix_oos.index)
        if len(common) > 10:
            ax.scatter(vix_oos.loc[common], v.loc[common],
                       s=6, alpha=0.5, color=c, label=sec)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("OOS: VIX level vs VRP (산점도)\n우하향 = VIX 높을 때 VRP 음전환", fontsize=9)
    ax.set_xlabel("VIX level", fontsize=8)
    ax.set_ylabel("VRP", fontsize=8)
    ax.legend(fontsize=5.5, ncol=3)
    ax.tick_params(labelsize=7)

    # ── (2,1) 주요 매크로 상관계수 (OOS, VIX level with VRP) ─────────────────
    ax = fig.add_subplot(gs[2, 1])
    corr_vix_lv = corr_df[(corr_df["macro"] == "VIX") & (corr_df["type"] == "level")]
    corr_vix_ch = corr_df[(corr_df["macro"] == "VIX") & (corr_df["type"] == "change")]
    corr_dxy    = corr_df[(corr_df["macro"] == "DXY") & (corr_df["type"] == "level")]
    corr_10y    = corr_df[(corr_df["macro"] == "US10Y") & (corr_df["type"] == "level")]

    datasets = [
        (corr_vix_lv, "VIX level",  "#C62828"),
        (corr_vix_ch, "VIX change", "#E65100"),
        (corr_dxy,    "DXY level",  "#1565C0"),
        (corr_10y,    "US10Y level","#2E7D32"),
    ]
    x = np.arange(len(SECTORS))
    width = 0.2
    for i, (df, label, color) in enumerate(datasets):
        if len(df) == 0:
            continue
        corrs = df.set_index("sector")["corr"].reindex(SECTORS).values
        ax.bar(x + (i - 1.5) * width, corrs, width,
               label=label, color=color, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(SECTORS, rotation=30, fontsize=8)
    ax.axhline(0,  color="black", linewidth=0.6)
    ax.axhline(0.3, color="gray", linewidth=0.5, linestyle="--", alpha=0.6)
    ax.axhline(-0.3,color="gray", linewidth=0.5, linestyle="--", alpha=0.6)
    ax.set_title("OOS 구간: 매크로 vs VRP 상관계수\n|corr|>0.3 = 의미 있는 연결", fontsize=9)
    ax.legend(fontsize=7)
    ax.set_ylabel("Correlation with VRP", fontsize=8)

    fig.suptitle("VRP 반전 원인 추적  |  OOS(2026) 분석", fontsize=13, y=1.01)
    out = FIGURES_DIR / "vrp_reversal_analysis.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sep = "=" * 62

    print(f"\n{sep}")
    print("  VRP 반전 원인 추적")
    print(sep)

    returns, capm_iv, rv, macro_lvl, macro_chg, vrp_full, iv_spread = load()

    # ── 1. 반전 시점 ───────────────────────────────────────────────────────────
    print(f"\n[1] 섹터별 VRP 반전 시점")
    rev_df = find_reversal_dates(vrp_full)
    print(f"  {'Sector':6s}  {'반전일':12s}  {'IS평균VRP':>10}  {'OOS평균VRP':>11}  {'상태':>6}")
    print("  " + "-" * 52)
    for sec in SECTORS:
        r = rev_df.loc[sec]
        rd = str(r["reversal_date"])[:10] if r["reversal_date"] else "없음(유지)"
        status = "반전" if r["vrp_oos_mean"] < 0 else "유지"
        print(f"  {sec:6s}  {rd:12s}  {r['vrp_is_mean']:>10.4f}  {r['vrp_oos_mean']:>11.4f}  {status:>6}")

    # ── 2. RV spike vs IV collapse ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("[2] 반전 원인: RV spike vs IV collapse")
    print(sep)
    decomp_df = rv_iv_decompose(capm_iv, rv)
    print(f"  {'Sector':6s}  {'delta_IV':>9}  {'delta_RV':>9}  {'VRP변화':>9}  주범")
    print("  " + "-" * 48)
    for sec in SECTORS:
        if sec not in decomp_df.index:
            continue
        r = decomp_df.loc[sec]
        print(f"  {sec:6s}  {r['delta_IV']:>+9.4f}  {r['delta_RV']:>+9.4f}  "
              f"{r['vrp_delta']:>+9.4f}  {r['culprit']}")

    # 전체 집계
    print(f"\n  집계:")
    culprit_counts = decomp_df["culprit"].value_counts()
    for culprit, cnt in culprit_counts.items():
        print(f"    {culprit}: {cnt}개 섹터")

    # ── 3. 매크로 트리거 ───────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("[3] 매크로 트리거 분석 (OOS 구간 상관계수)")
    print(sep)
    corr_df = macro_trigger_analysis(vrp_full, macro_lvl, macro_chg)

    # 가장 강한 매크로 연결 출력
    corr_lv = corr_df[corr_df["type"] == "level"].pivot(
        index="sector", columns="macro", values="corr")
    print(f"\n  VRP ~ 매크로 레벨 상관계수 (OOS)")
    print(corr_lv.reindex(SECTORS).round(3).to_string())

    # 전체 평균 — 어떤 매크로가 가장 연관?
    mean_corr = corr_lv.abs().mean().sort_values(ascending=False)
    print(f"\n  평균 |상관계수|: {mean_corr.to_dict()}")
    top_macro = mean_corr.index[0]
    print(f"  >> 가장 강한 트리거: {top_macro}")

    # VIX와의 상관 방향
    vix_corr = corr_lv["VIX"].reindex(SECTORS)
    neg_vix = vix_corr[vix_corr < -0.3]
    print(f"\n  VIX level과 음의 상관 (VIX 오르면 VRP 내려감):")
    for sec, c in neg_vix.sort_values().items():
        print(f"    {sec}: {c:.3f}")

    # ── 4. 베타 구조 변화 ──────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("[4] 베타 구조 변화 (IS vs OOS)")
    print(sep)
    beta_df = beta_shift_analysis(returns)
    print(f"  {'Sector':6s}  {'beta IS':>9}  {'beta OOS':>9}  {'shift':>8}  의미")
    print("  " + "-" * 45)
    for sec in SECTORS:
        if sec not in beta_df.index:
            continue
        r = beta_df.loc[sec]
        shift = r["beta_shift"]
        if np.isnan(shift):
            meaning = "N/A"
        elif abs(shift) > 0.2:
            meaning = "** 큰 변화 (CAPM IV 왜곡)"
        elif abs(shift) > 0.1:
            meaning = "* 중간 변화"
        else:
            meaning = "안정"
        print(f"  {sec:6s}  {r['beta_is']:>9.3f}  {r['beta_oos']:>9.3f}  "
              f"{shift:>+8.3f}  {meaning}")

    # ── 5. 종합 진단 ───────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("[5] 종합 진단")
    print(sep)

    reversal_secs = [s for s in SECTORS
                     if s in vrp_full.columns
                     and vrp_full.loc[vrp_full.index >= OOS_DT, s].mean() < 0]

    # 반전 섹터들의 공통점
    if reversal_secs:
        print(f"\n  VRP 음전환 섹터: {reversal_secs}")
        rev_beta = beta_df.loc[reversal_secs, "beta_shift"].mean()
        stable_beta = beta_df.loc[
            [s for s in SECTORS if s not in reversal_secs and s in beta_df.index],
            "beta_shift"].mean()
        print(f"  반전 섹터 평균 beta shift: {rev_beta:+.3f}")
        print(f"  유지 섹터 평균 beta shift: {stable_beta:+.3f}")

        rev_dIV = decomp_df.loc[reversal_secs, "delta_IV"].mean()
        rev_dRV = decomp_df.loc[reversal_secs, "delta_RV"].mean()
        print(f"  반전 섹터 평균 delta_IV: {rev_dIV:+.4f}")
        print(f"  반전 섹터 평균 delta_RV: {rev_dRV:+.4f}")

    # VIX 레벨 이벤트 확인
    vix_oos = macro_lvl["VIX"][macro_lvl.index >= OOS_DT].dropna()
    vix_peaks = vix_oos.nlargest(5)
    print(f"\n  OOS 구간 VIX 피크 상위 5개:")
    for dt, v in vix_peaks.items():
        print(f"    {str(dt)[:10]}: VIX={v:.1f}")

    vix_is = macro_lvl["VIX"][macro_lvl.index < OOS_DT].dropna()
    print(f"\n  VIX 통계 비교:")
    print(f"    IS  mean={vix_is.mean():.1f}  max={vix_is.max():.1f}  std={vix_is.std():.1f}")
    print(f"    OOS mean={vix_oos.mean():.1f}  max={vix_oos.max():.1f}  std={vix_oos.std():.1f}")

    # 핵심 결론
    print(f"\n  [핵심 결론]")
    print(f"  1. 반전 섹터 {reversal_secs}: delta_RV({rev_dRV:+.4f}) vs delta_IV({rev_dIV:+.4f})")
    main_cause = "RV_spike" if abs(rev_dRV) > abs(rev_dIV) else "IV_collapse"
    print(f"  2. 주범: {main_cause}")
    print(f"  3. 매크로 트리거: {top_macro} (mean |corr|={mean_corr[top_macro]:.3f})")
    print(f"  4. OOS VIX 변동성(std={vix_oos.std():.1f}) > IS(std={vix_is.std():.1f}): "
          f"{'YES - 레짐 변화' if vix_oos.std() > vix_is.std() * 1.3 else 'NO - 레짐 유사'}")

    # ── 시각화 ────────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("[6] 시각화...")
    plot_reversal(vrp_full, decomp_df, beta_df, macro_lvl, corr_df)

    print(f"\n{sep}")
    print("  완료.")
    print(sep)

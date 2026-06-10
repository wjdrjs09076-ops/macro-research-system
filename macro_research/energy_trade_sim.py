"""
energy_trade_sim.py - 에너지 기업 개별주 트레이딩 시뮬레이션

신호 정의 (XLE 섹터 레벨 신호를 개별주에 적용):
  S1: XLE 롤링 베타(30D) < -0.05 이상 3일 연속  → LONG
      (에너지가 시장과 역방향 = 공급 충격 구조 확인)
  S2: VRP_XLE < 0 이상 3일 연속                 → LONG
      (옵션이 실제 변동성보다 싸게 거래됨 = 변동성 폭발 예고)
  S3: S1 AND S2 동시 충족                        → STRONG LONG

포지션:
  진입: 신호 발생 다음날 시가 (t+1)
  청산: 신호 소멸 다음날 시가
  포지션: 100% 롱 (레버리지 없음)
  거래비용: 미국 0.05%, 한국 0.35% (매도세 + 수수료)

대상:
  미국: XOM, CVX, COP, OXY, SLB, EOG
  한국: 010950.KS (S-Oil), 096770.KS (SK이노베이션),
        078930.KS (GS Holdings), 036460.KS (한국가스공사),
        267250.KS (HD현대중공업/에너지 관련)
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
import yfinance as yf

from config import OUTPUT_DIR, FIGURES_DIR, OOS_START

OOS_DT = pd.Timestamp(OOS_START)
START  = "2025-01-01"
END    = "2026-05-29"

# 거래비용
TC_US = 0.0005   # 0.05% per trade (매수+매도 합산)
TC_KR = 0.0035   # 0.35% (증권거래세 0.20% + 수수료 0.15%)

US_STOCKS = {
    "XOM": "ExxonMobil",
    "CVX": "Chevron",
    "COP": "ConocoPhillips",
    "OXY": "Occidental Petroleum",
    "SLB": "SLB (Schlumberger)",
    "EOG": "EOG Resources",
}

KR_STOCKS = {
    "010950.KS": "S-Oil",
    "096770.KS": "SK이노베이션",
    "078930.KS": "GS Holdings",
    "036460.KS": "한국가스공사",
    "267250.KS": "HD현대중공업",
}

# 주요 이벤트
EVENTS = [
    ("2025-06-22", "미국 이란 공습",     "#C62828"),
    ("2026-02-28", "호르무즈 완전 봉쇄", "#B71C1C"),
    ("2026-03-04", "IRGC 공식 폐쇄",    "#E65100"),
    ("2026-04-07", "유가 $138 최고",     "#FF8F00"),
    ("2026-04-08", "휴전 합의",          "#2E7D32"),
    ("2026-05-04", "휴전 붕괴",          "#C62828"),
]


# ---------------------------------------------------------------------------
# 데이터 수집
# ---------------------------------------------------------------------------

def fetch_prices(tickers: list[str]) -> pd.DataFrame:
    """yfinance에서 조정종가 수집. 실패 종목은 제외."""
    result = {}
    for tkr in tickers:
        try:
            raw = yf.download(tkr, start=START, end=END,
                              auto_adjust=True, progress=False)["Close"]
            if isinstance(raw, pd.DataFrame):
                raw = raw.iloc[:, 0]
            raw = raw.squeeze().dropna()
            if len(raw) > 50:
                result[tkr] = raw
                print(f"  OK  {tkr:15s}  ({len(raw)}일)")
            else:
                print(f"  SKIP {tkr} (데이터 부족: {len(raw)}일)")
        except Exception as e:
            print(f"  FAIL {tkr}: {e}")
    if not result:
        return pd.DataFrame()
    return pd.DataFrame(result).sort_index()


def load_signals() -> tuple[pd.Series, pd.Series]:
    """XLE 베타 신호 + VRP 신호 로드."""
    returns = pd.read_parquet(OUTPUT_DIR / "sector_returns.parquet")
    capm_iv = pd.read_parquet(OUTPUT_DIR / "capm_iv.parquet")
    rv      = pd.read_parquet(OUTPUT_DIR / "realized_vol.parquet")

    # XLE 롤링 베타 30D
    spy = returns["SPY"]
    xle = returns["XLE"]
    window = 30
    b_vals, dates = [], []
    for end in range(window, len(returns) + 1):
        r_s = xle.iloc[end - window:end].values
        r_m = spy.iloc[end - window:end].values
        mask = ~(np.isnan(r_s) | np.isnan(r_m))
        if mask.sum() < 15:
            b_vals.append(np.nan)
        else:
            cov = np.cov(r_s[mask], r_m[mask])
            b_vals.append(cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else np.nan)
        dates.append(returns.index[end - 1])
    beta_xle = pd.Series(b_vals, index=dates, name="beta_XLE")

    # XLE VRP
    vrp_xle = (capm_iv["XLE"] - rv["XLE"].reindex(capm_iv.index)).rename("vrp_XLE")

    return beta_xle, vrp_xle


def build_signal_flags(beta: pd.Series, vrp: pd.Series,
                        consec: int = 3) -> pd.DataFrame:
    """
    S1: beta < -0.05 연속 consec일
    S2: vrp < 0 연속 consec일
    S3: S1 AND S2
    """
    b_neg = (beta < -0.05).astype(int)
    v_neg = (vrp  < 0.00).astype(int)

    def _consec(s, n):
        return s.rolling(n).sum() >= n

    common = beta.index.intersection(vrp.index)
    s1 = _consec(b_neg.reindex(common), consec).astype(int)
    s2 = _consec(v_neg.reindex(common), consec).astype(int)
    s3 = ((s1 == 1) & (s2 == 1)).astype(int)

    return pd.DataFrame({"S1_beta": s1, "S2_vrp": s2, "S3_combo": s3},
                        index=common)


# ---------------------------------------------------------------------------
# 백테스트 엔진
# ---------------------------------------------------------------------------

def backtest_single(prices: pd.Series, signal: pd.Series,
                     tc: float = 0.0005) -> dict:
    """
    prices  : 조정종가 (일간)
    signal  : 0/1 포지션 신호 (1=보유, 0=현금)
    tc      : 거래비용 (매매 1회 편도 기준)

    진입/청산은 신호 다음날 시가 근사 (종가 사용, 슬리피지 미반영)
    """
    rets = np.log(prices / prices.shift(1)).dropna()
    sig  = signal.reindex(rets.index).ffill().fillna(0)

    # 포지션 변화 시 거래비용 차감
    trades = sig.diff().abs().fillna(0)
    port_rets = sig.shift(1).fillna(0) * rets - trades * tc

    cum  = port_rets.cumsum()
    ann  = port_rets.mean() * 252
    vol  = port_rets.std() * np.sqrt(252)
    sh   = ann / vol if vol > 0 else np.nan
    mdd  = (cum - cum.cummax()).min()
    n_trades = int((trades > 0).sum())

    # OOS 전용 성과
    oos_rets = port_rets[port_rets.index >= OOS_DT]
    oos_ann  = oos_rets.mean() * 252 if len(oos_rets) > 5 else np.nan
    oos_vol  = oos_rets.std() * np.sqrt(252) if len(oos_rets) > 5 else np.nan
    oos_sh   = oos_ann / oos_vol if (oos_vol and oos_vol > 0) else np.nan
    oos_cum  = oos_rets.cumsum()
    oos_mdd  = (oos_cum - oos_cum.cummax()).min() if len(oos_rets) > 5 else np.nan

    # BnH 비교
    bnh_rets  = rets
    bnh_ann   = bnh_rets.mean() * 252
    bnh_vol   = bnh_rets.std() * np.sqrt(252)
    bnh_sh    = bnh_ann / bnh_vol if bnh_vol > 0 else np.nan
    bnh_oos   = bnh_rets[bnh_rets.index >= OOS_DT]
    bnh_oos_ann = bnh_oos.mean() * 252 if len(bnh_oos) > 5 else np.nan

    return {
        "port_rets":  port_rets,
        "cum":        cum,
        "ann":        ann, "vol": vol, "sharpe": sh, "mdd": mdd,
        "n_trades":   n_trades,
        "oos_ann":    oos_ann, "oos_vol": oos_vol,
        "oos_sharpe": oos_sh, "oos_mdd": oos_mdd,
        "bnh_ann":    bnh_ann, "bnh_sharpe": bnh_sh,
        "bnh_oos_ann": bnh_oos_ann,
    }


def run_all_backtests(price_df: pd.DataFrame, signals: pd.DataFrame,
                       tc: float, label: str) -> pd.DataFrame:
    rows = []
    for ticker in price_df.columns:
        prices = price_df[ticker].dropna()
        for sig_name in ["S1_beta", "S2_vrp", "S3_combo"]:
            sig = signals[sig_name].reindex(prices.index).ffill().fillna(0)
            r = backtest_single(prices, sig, tc)
            name = US_STOCKS.get(ticker, KR_STOCKS.get(ticker, ticker))
            rows.append({
                "ticker":    ticker,
                "name":      name,
                "signal":    sig_name,
                "market":    label,
                **{k: v for k, v in r.items()
                   if k not in ("port_rets", "cum")},
                "_port_rets": r["port_rets"],
                "_cum":       r["cum"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------

def plot_results(results: pd.DataFrame, signals: pd.DataFrame,
                  beta: pd.Series, vrp: pd.Series):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── 그림 1: 미국 에너지주 누적수익 (S3 신호) ─────────────────────────────
    us_s3 = results[(results["market"] == "US") & (results["signal"] == "S3_combo")]
    kr_s3 = results[(results["market"] == "KR") & (results["signal"] == "S3_combo")]

    fig = plt.figure(figsize=(20, 22))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)

    def _add_events(ax):
        ylo, yhi = ax.get_ylim()
        for dt, lb, c in EVENTS:
            ax.axvline(pd.Timestamp(dt), color=c, linewidth=0.9,
                       linestyle="--", alpha=0.7)
            ax.text(pd.Timestamp(dt), yhi * 0.95, lb,
                    rotation=90, fontsize=5.5, color=c,
                    va="top", ha="right", alpha=0.85)

    # (0,0) 미국 에너지주 S3 누적수익
    ax = fig.add_subplot(gs[0, 0])
    colors = plt.cm.tab10(np.linspace(0, 1, len(us_s3)))
    for (_, row), c in zip(us_s3.iterrows(), colors):
        row["_cum"].mul(100).plot(ax=ax, label=f"{row['ticker']} ({row['name'][:8]})",
                                   color=c, linewidth=1.1)
    ax.axvline(OOS_DT, color="black", linewidth=1.5, linestyle=":")
    ax.axhline(0, color="black", linewidth=0.5)
    _add_events(ax)
    ax.set_title("미국 에너지주 누적수익 (S3 복합신호)\n거래비용 0.05% 반영",
                 fontsize=9)
    ax.legend(fontsize=6.5, ncol=2)
    ax.set_ylabel("누적 로그수익 (%)", fontsize=8)
    ax.tick_params(labelsize=7)

    # (0,1) 한국 에너지주 S3 누적수익
    ax = fig.add_subplot(gs[0, 1])
    colors_kr = plt.cm.Set2(np.linspace(0, 1, max(len(kr_s3), 1)))
    for (_, row), c in zip(kr_s3.iterrows(), colors_kr):
        row["_cum"].mul(100).plot(ax=ax, label=f"{row['ticker'].split('.')[0]} ({row['name'][:8]})",
                                   color=c, linewidth=1.1)
    ax.axvline(OOS_DT, color="black", linewidth=1.5, linestyle=":")
    ax.axhline(0, color="black", linewidth=0.5)
    _add_events(ax)
    ax.set_title("한국 에너지주 누적수익 (S3 복합신호)\n거래비용 0.35% 반영",
                 fontsize=9)
    ax.legend(fontsize=6.5, ncol=2)
    ax.set_ylabel("누적 로그수익 (%)", fontsize=8)
    ax.tick_params(labelsize=7)

    # (1,0~1) 신호 가시화 (베타 + VRP + S1/S2/S3)
    ax = fig.add_subplot(gs[1, :])
    ax2 = ax.twinx()
    beta.plot(ax=ax, label="XLE beta (30D)", color="#E65100", linewidth=1.1)
    ax.axhline(0,     color="black",  linewidth=0.8)
    ax.axhline(-0.05, color="#C62828", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.fill_between(beta.index, beta, -0.05,
                    where=(beta < -0.05), alpha=0.2, color="#C62828",
                    label="S1 발동 구간")
    ax.set_ylabel("XLE beta", fontsize=8, color="#E65100")
    ax.tick_params(axis="y", labelcolor="#E65100", labelsize=7)

    vrp.rolling(3).mean().plot(ax=ax2, label="VRP_XLE (3D MA)",
                                color="#1565C0", linewidth=1.0, linestyle="--")
    ax2.axhline(0, color="#1565C0", linewidth=0.6, linestyle=":")
    ax2.set_ylabel("VRP_XLE", fontsize=8, color="#1565C0")
    ax2.tick_params(axis="y", labelcolor="#1565C0", labelsize=7)
    ax.axvline(OOS_DT, color="black", linewidth=1.5, linestyle=":")
    _add_events(ax)
    lines1, lb1 = ax.get_legend_handles_labels()
    lines2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lb1 + lb2, fontsize=7, loc="upper left")
    ax.set_title("신호 시계열: XLE beta + VRP\n빨간 음영 = S1 발동, OOS 시작(검정점선)",
                 fontsize=9)
    ax.tick_params(axis="x", labelsize=7)

    # (2,0) 신호별 OOS CAGR 비교 — 미국
    ax = fig.add_subplot(gs[2, 0])
    sig_labels = {"S1_beta": "S1: beta<-0.05", "S2_vrp": "S2: VRP<0",
                  "S3_combo": "S3: 복합"}
    us_res = results[results["market"] == "US"]
    x = np.arange(len(US_STOCKS))
    width = 0.25
    for i, (sig_key, sig_lab) in enumerate(sig_labels.items()):
        sub = us_res[us_res["signal"] == sig_key].set_index("ticker")
        vals = [sub.loc[t, "oos_ann"] * 100 if t in sub.index else 0
                for t in US_STOCKS]
        ax.bar(x + (i-1)*width, vals, width, label=sig_lab, alpha=0.85)
    # BnH
    bnh_vals = [us_res[us_res["signal"] == "S1_beta"].set_index("ticker").loc[t, "bnh_oos_ann"] * 100
                if t in us_res.set_index("ticker").index else 0
                for t in US_STOCKS]
    ax.plot(x, bnh_vals, "ko--", markersize=5, linewidth=0.8, label="Buy&Hold OOS")
    ax.set_xticks(x)
    ax.set_xticklabels(list(US_STOCKS.keys()), fontsize=9)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("미국 에너지주 OOS CAGR by 신호\n(거래비용 반영, 연환산)", fontsize=9)
    ax.set_ylabel("OOS CAGR (%)", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    # (2,1) 신호별 OOS CAGR — 한국
    ax = fig.add_subplot(gs[2, 1])
    kr_res = results[results["market"] == "KR"]
    kr_tickers = [t for t in KR_STOCKS if t in kr_res["ticker"].values]
    if kr_tickers:
        x = np.arange(len(kr_tickers))
        for i, (sig_key, sig_lab) in enumerate(sig_labels.items()):
            sub = kr_res[kr_res["signal"] == sig_key].set_index("ticker")
            vals = [sub.loc[t, "oos_ann"] * 100 if t in sub.index else 0
                    for t in kr_tickers]
            ax.bar(x + (i-1)*width, vals, width, label=sig_lab, alpha=0.85)
        bnh_kr = [kr_res[kr_res["signal"] == "S1_beta"].set_index("ticker").loc[t, "bnh_oos_ann"] * 100
                  if t in kr_res.set_index("ticker").index else 0
                  for t in kr_tickers]
        ax.plot(x, bnh_kr, "ko--", markersize=5, linewidth=0.8, label="Buy&Hold OOS")
        ax.set_xticks(x)
        ax.set_xticklabels([t.split(".")[0] for t in kr_tickers], fontsize=9)
        ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("한국 에너지주 OOS CAGR by 신호\n(거래비용 0.35% 반영)", fontsize=9)
    ax.set_ylabel("OOS CAGR (%)", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    # (3,0~1) 포지션 보유 기간 비교 (히트맵)
    ax = fig.add_subplot(gs[3, :])
    all_tickers = list(US_STOCKS.keys()) + [t.split(".")[0] for t in KR_STOCKS]
    sig_order   = ["S1_beta", "S2_vrp", "S3_combo"]
    heatmap_data = np.zeros((len(sig_order), len(all_tickers)))
    heatmap_cagr = np.zeros((len(sig_order), len(all_tickers)))

    for i, sig_key in enumerate(sig_order):
        sub_us = results[(results["market"] == "US") & (results["signal"] == sig_key)].set_index("ticker")
        sub_kr = results[(results["market"] == "KR") & (results["signal"] == sig_key)].set_index("ticker")
        for j, tkr in enumerate(list(US_STOCKS.keys())):
            if tkr in sub_us.index:
                heatmap_cagr[i, j] = sub_us.loc[tkr, "oos_ann"] * 100
        for j, tkr in enumerate(KR_STOCKS.keys()):
            jj = len(US_STOCKS) + list(KR_STOCKS.keys()).index(tkr)
            if tkr in sub_kr.index:
                heatmap_cagr[i, jj] = sub_kr.loc[tkr, "oos_ann"] * 100

    im = ax.imshow(heatmap_cagr, cmap="RdYlGn", aspect="auto",
                   vmin=-50, vmax=150)
    ax.set_xticks(range(len(all_tickers)))
    ax.set_xticklabels(all_tickers, rotation=30, fontsize=8)
    ax.set_yticks(range(len(sig_order)))
    ax.set_yticklabels(["S1: beta<-0.05", "S2: VRP<0", "S3: 복합"], fontsize=9)
    for i in range(len(sig_order)):
        for j in range(len(all_tickers)):
            ax.text(j, i, f"{heatmap_cagr[i,j]:.0f}%",
                    ha="center", va="center", fontsize=7.5,
                    color="white" if abs(heatmap_cagr[i,j]) > 60 else "black")
    plt.colorbar(im, ax=ax, label="OOS CAGR (%)")
    ax.set_title("OOS(2026) CAGR 히트맵 — 신호 × 종목\n녹색=수익, 빨강=손실",
                 fontsize=10)

    fig.suptitle(
        "에너지 기업 트레이딩 시뮬레이션\n"
        "신호: XLE 섹터 beta 반전 + VRP 음전환 → 개별 에너지주 LONG",
        fontsize=12, y=1.01
    )
    out = FIGURES_DIR / "energy_trade_sim.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sep = "=" * 66

    print(f"\n{sep}")
    print("  에너지 기업 트레이딩 시뮬레이션")
    print(f"  신호: XLE beta 반전 + VRP 음전환 → 롱 포지션")
    print(sep)

    # ── 신호 구성 ─────────────────────────────────────────────────────────────
    print("\n[1] 신호 구성 중...")
    beta_xle, vrp_xle = load_signals()
    signals = build_signal_flags(beta_xle, vrp_xle, consec=3)

    # 신호 발동 구간 출력
    for sig_name in ["S1_beta", "S2_vrp", "S3_combo"]:
        s = signals[sig_name]
        oos_s = s[s.index >= OOS_DT]
        print(f"  {sig_name}: OOS 포지션 보유일 {oos_s.sum()}일 / "
              f"총 {len(oos_s)}일 ({oos_s.mean()*100:.1f}%)")

    # ── 데이터 수집 ───────────────────────────────────────────────────────────
    print(f"\n[2] 미국 에너지주 가격 수집...")
    us_prices = fetch_prices(list(US_STOCKS.keys()))

    print(f"\n[3] 한국 에너지주 가격 수집...")
    kr_prices = fetch_prices(list(KR_STOCKS.keys()))

    # ── 백테스트 ──────────────────────────────────────────────────────────────
    print(f"\n[4] 백테스트 실행...")
    all_results = []
    if len(us_prices) > 0:
        r_us = run_all_backtests(us_prices, signals, TC_US, "US")
        all_results.append(r_us)
    if len(kr_prices) > 0:
        r_kr = run_all_backtests(kr_prices, signals, TC_KR, "KR")
        all_results.append(r_kr)

    if not all_results:
        print("  데이터 없음 — 종료")
        exit()

    results = pd.concat(all_results, ignore_index=True)

    # ── 결과 출력 ─────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("[5] 성과 요약")
    print(sep)

    for market, tc_label in [("US", "0.05%"), ("KR", "0.35%")]:
        sub = results[results["market"] == market]
        if sub.empty:
            continue
        print(f"\n  [{market} 에너지주]  거래비용 {tc_label}/회")
        print(f"  {'Ticker':14s}  {'Signal':10s}  "
              f"{'전체CAGR':>9}  {'전체Sharpe':>10}  "
              f"{'OOS CAGR':>9}  {'OOS Sharpe':>10}  "
              f"{'OOS MDD':>8}  {'BnH OOS':>9}  {'거래수':>5}")
        print("  " + "-" * 95)
        for _, r in sub.sort_values(["ticker","signal"]).iterrows():
            def _fmt(v):
                return f"{v*100:+.1f}%" if not (v is None or (isinstance(v, float) and np.isnan(v))) else "  N/A"
            def _fmts(v):
                return f"{v:+.2f}" if not (v is None or (isinstance(v, float) and np.isnan(v))) else "  N/A"
            print(f"  {r['ticker']:14s}  {r['signal']:10s}  "
                  f"{_fmt(r['ann']):>9}  {_fmts(r['sharpe']):>10}  "
                  f"{_fmt(r['oos_ann']):>9}  {_fmts(r['oos_sharpe']):>10}  "
                  f"{_fmt(r['oos_mdd']):>8}  {_fmt(r['bnh_oos_ann']):>9}  "
                  f"{r['n_trades']:>5}")

    # ── 베스트 전략 요약 ──────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("[6] OOS 기준 최고 성과 TOP 10")
    print(sep)
    top10 = (results[results["signal"] == "S3_combo"]
             .dropna(subset=["oos_ann"])
             .sort_values("oos_ann", ascending=False)
             .head(10))
    print(f"  {'Ticker':14s}  {'Name':20s}  {'Market':6s}  "
          f"{'OOS CAGR':>9}  {'OOS Sharpe':>10}  {'OOS MDD':>8}  {'BnH OOS':>9}")
    print("  " + "-" * 82)
    for _, r in top10.iterrows():
        def _fmt(v):
            return f"{v*100:+.1f}%" if not (v is None or (isinstance(v, float) and np.isnan(v))) else "  N/A"
        def _fmts(v):
            return f"{v:+.2f}" if not (v is None or (isinstance(v, float) and np.isnan(v))) else "  N/A"
        print(f"  {r['ticker']:14s}  {r['name']:20s}  {r['market']:6s}  "
              f"{_fmt(r['oos_ann']):>9}  {_fmts(r['oos_sharpe']):>10}  "
              f"{_fmt(r['oos_mdd']):>8}  {_fmt(r['bnh_oos_ann']):>9}")

    # ── S3 평균 성과 ──────────────────────────────────────────────────────────
    print(f"\n{sep}")
    s3 = results[results["signal"] == "S3_combo"].dropna(subset=["oos_ann"])
    print("[7] S3(복합신호) 평균 성과 요약")
    for market in ["US", "KR", None]:
        sub = s3[s3["market"] == market] if market else s3
        label = market if market else "전체"
        if sub.empty:
            continue
        print(f"  {label:6s}  평균OOS CAGR={sub['oos_ann'].mean()*100:+.1f}%  "
              f"평균OOS Sharpe={sub['oos_sharpe'].mean():+.2f}  "
              f"평균BnH OOS={sub['bnh_oos_ann'].mean()*100:+.1f}%  "
              f"(n={len(sub)}종목)")

    # ── 시각화 ────────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("[8] 시각화...")
    plot_results(results, signals, beta_xle, vrp_xle)

    # CSV 저장
    save_cols = [c for c in results.columns if not c.startswith("_")]
    results[save_cols].to_csv(OUTPUT_DIR / "energy_trade_results.csv", index=False)
    print(f"  Saved: energy_trade_results.csv")

    print(f"\n{sep}")
    print("  완료.")
    print(sep)

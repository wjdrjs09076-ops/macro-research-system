"""
historical_crisis_test.py

프레임워크 일반화 검증:
동일 신호(섹터 beta 체제전환 + VRP 음전환)를 4개 역사적 위기에 적용

  닷컴버블  (2000) -> XLK -> SHORT: tech가 시장 하락 증폭 (beta > 1.5 + VRP < 0)
  금융위기  (2008) -> XLF -> SHORT: 금융주가 위기 진원지 (beta > 1.5 + VRP < 0)
  COVID-19  (2020) -> XLV -> LONG : 헬스케어 디커플링 상승  (beta < -0.05 + VRP < 0)
  호르무즈  (2025) -> XLE -> LONG : 에너지 공급충격 수혜    (beta < -0.05 + VRP < 0)

VRP는 GARCH 대신 EWMA(lambda=0.94) 기반 → 수십 년 크로스-데케이드 일관성 확보

출력:
  output/figures/historical_crisis_test.png
  output/crisis_results.csv
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).parent
OUTPUT_DIR  = BASE_DIR / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 위기-섹터 매핑
# ---------------------------------------------------------------------------
CRISES = {
    "dotcom": {
        "label"       : "닷컴버블 붕괴 (2000-2002)",
        "event_date"  : "2000-03-10",          # NASDAQ 5048 고점
        "window"      : ("1998-01-02", "2002-12-31"),
        "oos_start"   : "2000-03-10",          # 고점부터 = SHORT 신호 구간
        "etf"         : "XLK",
        "direction"   : "SHORT",
        "beta_thresh" : 1.5,
        "us_stocks"   : {
            "MSFT": "Microsoft",
            "INTC": "Intel",
            "CSCO": "Cisco",
            "ORCL": "Oracle",
            "AMZN": "Amazon",
        },
        "kr_stocks"   : {},                    # 당시 한국 인터넷주 데이터 없음
        "event_label" : "NASDAQ 고점\n2000-03-10",
        "color"       : "#d62728",
    },
    "gfc": {
        "label"       : "글로벌 금융위기 (2007-2009)",
        "event_date"  : "2008-09-15",          # 리먼 파산
        "window"      : ("2006-01-02", "2010-12-31"),
        "oos_start"   : "2008-09-15",          # 리먼부터 = 위기 심화 구간
        "etf"         : "XLF",
        "direction"   : "SHORT",
        "beta_thresh" : 1.5,
        "us_stocks"   : {
            "GS" : "Goldman Sachs",
            "MS" : "Morgan Stanley",
            "BAC": "Bank of America",
            "C"  : "Citigroup",
            "JPM": "JPMorgan",
        },
        "kr_stocks"   : {
            "055550.KS": "신한지주",
            "086790.KS": "하나금융",
        },
        "event_label" : "리먼 파산\n2008-09-15",
        "color"       : "#d62728",
    },
    "covid": {
        "label"       : "COVID-19 팬데믹 (2020-2021)",
        "event_date"  : "2020-02-20",          # SPY 최고점
        "window"      : ("2019-01-02", "2021-12-31"),
        "oos_start"   : "2020-02-20",          # 충격 시작부터 = LONG 수혜 구간
        "etf"         : "XLV",
        "direction"   : "LONG",
        "beta_thresh" : -0.05,
        "us_stocks"   : {
            "PFE"  : "Pfizer",
            "MRNA" : "Moderna",
            "JNJ"  : "Johnson & Johnson",
            "ABBV" : "AbbVie",
            "REGN" : "Regeneron",
        },
        "kr_stocks"   : {
            "207940.KS": "삼성바이오로직스",
            "068270.KS": "셀트리온",
        },
        "event_label" : "SPY 고점\n2020-02-20",
        "color"       : "#2ca02c",
    },
    "hormuz": {
        "label"       : "호르무즈 봉쇄 (2025-2026)",
        "event_date"  : "2025-06-01",
        "window"      : ("2025-01-02", "2026-05-29"),
        "oos_start"   : "2026-01-01",
        "etf"         : "XLE",
        "direction"   : "LONG",
        "beta_thresh" : -0.05,
        "us_stocks"   : {
            "XOM": "ExxonMobil",
            "CVX": "Chevron",
            "SLB": "SLB",
            "OXY": "Occidental",
            "COP": "ConocoPhillips",
        },
        "kr_stocks"   : {
            "267250.KS": "HD현대중공업",
            "078930.KS": "GS Holdings",
        },
        "event_label" : "Operation\nMidnight Hammer",
        "color"       : "#2ca02c",
    },
}

TC_US    = 0.0005
TC_KR    = 0.0035
BETA_WIN = 30
RV_WIN   = 20
EWMA_SPAN = 32   # lambda=0.94 -> span = 2/(1-0.94)-1 ≈ 32
CONSEC   = 3

# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------

def _fetch_close(tickers: list, start: str, end: str) -> pd.DataFrame:
    """yfinance batch download, returns Close DataFrame"""
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw
    # single ticker case: raw is Series with price name = ticker
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    return close


def fetch_all(stock_tickers: list, etf: str, start: str, end: str) -> dict:
    """Returns dict of clean price Series keyed by ticker."""
    all_t = list(set(stock_tickers + [etf, "SPY", "^VIX"]))
    close  = _fetch_close(all_t, start, end)
    out = {}
    for t in all_t:
        if t in close.columns:
            s = close[t].dropna()
            if len(s) > 50:
                out[t] = s
    return out

# ---------------------------------------------------------------------------
# 신호 계산
# ---------------------------------------------------------------------------

def rolling_beta(sec_ret: pd.Series, spy_ret: pd.Series,
                 window: int = BETA_WIN) -> pd.Series:
    idx = sec_ret.index.intersection(spy_ret.index)
    s   = sec_ret.loc[idx]
    m   = spy_ret.loc[idx]
    cov = s.rolling(window).cov(m)
    var = m.rolling(window).var()
    return (cov / var).dropna()


def ewma_std(series: pd.Series, span: int = EWMA_SPAN) -> pd.Series:
    """Exponentially weighted standard deviation (lambda = 1 - 2/(span+1))"""
    return series.ewm(span=span, min_periods=20).std()


def compute_vrp(sec_ret: pd.Series, spy_ret: pd.Series,
                vix_series: pd.Series) -> tuple:
    """
    VRP = CAPM_IV_ann - RV_ann
    CAPM_IV uses 63D rolling beta + EWMA idiosyncratic vol.
    Both in annualized units.
    """
    idx = sec_ret.index.intersection(spy_ret.index)
    s   = sec_ret.loc[idx]
    m   = spy_ret.loc[idx]

    # 63D beta (more stable for decomposition)
    b63 = rolling_beta(s, m, window=63).reindex(idx).ffill()

    # EWMA idio vol
    residuals  = s - b63 * m
    idio_daily = ewma_std(residuals)
    idio_ann   = idio_daily * np.sqrt(252)

    # VIX -> annualized market vol (VIX is already %)
    vix_ann = vix_series.reindex(idx).ffill() / 100.0

    capm_iv = np.sqrt((b63 * vix_ann) ** 2 + idio_ann ** 2)
    rv_ann  = s.rolling(RV_WIN).std() * np.sqrt(252)
    vrp     = (capm_iv - rv_ann).dropna()

    return vrp, capm_iv, rv_ann


def build_signal(beta_30: pd.Series, vrp: pd.Series,
                 direction: str, beta_thresh: float,
                 consec: int = CONSEC) -> pd.Series:
    """
    LONG : fire when beta < thresh AND vrp < 0 for consec consecutive days
    SHORT: fire when beta > thresh AND vrp < 0 for consec consecutive days
    Returns boolean series shifted +1 (trade on next open)
    """
    idx = beta_30.index.intersection(vrp.index)
    b   = beta_30.reindex(idx)
    v   = vrp.reindex(idx)

    if direction == "LONG":
        raw = (b < beta_thresh) & (v < 0)
    else:
        raw = (b > beta_thresh) & (v < 0)

    confirmed = raw.rolling(consec).sum().ge(consec)
    return confirmed.shift(1).fillna(False)


# ---------------------------------------------------------------------------
# 백테스트
# ---------------------------------------------------------------------------

def backtest(price_s: pd.Series, signal: pd.Series,
             direction: str, tc: float,
             oos_start: str = None) -> dict:
    ret = price_s.pct_change().dropna()
    sig = signal.reindex(ret.index).fillna(False)

    # position: +1 LONG, -1 SHORT
    pos        = sig.astype(float) * (1.0 if direction == "LONG" else -1.0)
    trade_cost = pos.diff().abs() * tc
    net_ret    = pos * ret - trade_cost

    cum = (1 + net_ret).cumprod()
    bnh = (1 + ret).cumprod()

    def _stats(r, c, b):
        n = len(r)
        if n < 5 or c.iloc[-1] <= 0:
            return dict(cagr=np.nan, sharpe=np.nan, mdd=np.nan,
                        bnh=np.nan, n_trade=0, hit_rate=np.nan)
        cagr    = c.iloc[-1] ** (252 / n) - 1
        bnh_r   = b.iloc[-1] ** (252 / n) - 1
        sharpe  = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
        roll_max = c.cummax()
        mdd     = ((c - roll_max) / roll_max).min()
        n_trade = int(pos.diff().abs().sum() / 2)
        # 5D forward hit rate from signal fires
        sig_on  = sig[sig == True]
        hits = []
        for dt in sig_on.index:
            loc = ret.index.get_loc(dt)
            fwd = ret.iloc[loc + 1: loc + 6].sum()
            hits.append((fwd > 0) if direction == "LONG" else (fwd < 0))
        hit_rate = np.mean(hits) if hits else np.nan
        return dict(cagr=cagr, sharpe=sharpe, mdd=mdd, bnh=bnh_r,
                    n_trade=n_trade, hit_rate=hit_rate)

    result = {"total": _stats(net_ret, cum, bnh), "cum": cum, "bnh": bnh}

    if oos_start:
        oos_dt  = pd.Timestamp(oos_start)
        r_oos   = net_ret[net_ret.index >= oos_dt]
        c_oos   = (1 + r_oos).cumprod()
        b_oos   = (1 + ret[ret.index >= oos_dt]).cumprod()
        result["oos"] = _stats(r_oos, c_oos, b_oos)

    return result


# ---------------------------------------------------------------------------
# 위기별 실행
# ---------------------------------------------------------------------------

def run_crisis(name: str, cfg: dict) -> dict | None:
    start, end = cfg["window"]
    oos_start  = cfg["oos_start"]
    direction  = cfg["direction"]
    etf        = cfg["etf"]

    print(f"\n{'─'*60}")
    print(f"  {cfg['label']}")
    print(f"  신호: {etf} beta {'<' if direction=='LONG' else '>'} {cfg['beta_thresh']}"
          f"  AND  VRP < 0  ->  {direction}")
    print(f"{'─'*60}")

    # 전체 티커 다운로드
    all_stocks = list(cfg["us_stocks"]) + list(cfg["kr_stocks"])
    prices     = fetch_all(all_stocks, etf, start, end)

    if "SPY" not in prices:
        print("  [ERROR] SPY 데이터 없음 - 건너뜀")
        return None
    if "^VIX" not in prices:
        print("  [ERROR] VIX 데이터 없음 - 건너뜀")
        return None
    if etf not in prices:
        print(f"  [ERROR] {etf} 데이터 없음 - 건너뜀")
        return None

    spy_ret = prices["SPY"].pct_change().dropna()
    vix     = prices["^VIX"]
    etf_ret = prices[etf].pct_change().dropna()

    # 섹터 신호 계산
    beta_30       = rolling_beta(etf_ret, spy_ret.reindex(etf_ret.index).dropna())
    vrp, capm_iv, rv = compute_vrp(etf_ret,
                                   spy_ret.reindex(etf_ret.index).dropna(),
                                   vix)
    signal        = build_signal(beta_30, vrp, direction, cfg["beta_thresh"])

    oos_dt   = pd.Timestamp(oos_start)
    n_sig    = int(signal.sum())
    n_sig_oos = int(signal[signal.index >= oos_dt].sum())
    total_d  = len(signal)
    oos_d    = len(signal[signal.index >= oos_dt])
    print(f"  신호 발생: 전체 {n_sig}일/{total_d}일 ({100*n_sig/total_d:.1f}%)"
          f"  |  OOS {n_sig_oos}일/{oos_d}일 ({100*n_sig_oos/max(oos_d,1):.1f}%)")

    # 개별 종목 백테스트
    rows = []
    for tkr, stock_name in {**cfg["us_stocks"], **cfg["kr_stocks"]}.items():
        if tkr not in prices:
            print(f"  SKIP {tkr:<14} (데이터 부족)")
            continue
        market = "KR" if tkr.endswith(".KS") else "US"
        tc     = TC_KR if market == "KR" else TC_US
        res    = backtest(prices[tkr], signal, direction, tc, oos_start)
        tot    = res["total"]
        oos    = res.get("oos", {})
        print(f"  {tkr:<14} | 전체: CAGR={tot['cagr']*100:+.1f}% Sharpe={tot['sharpe']:.2f}"
              f" | OOS: CAGR={oos.get('cagr',np.nan)*100:+.1f}%"
              f" Sharpe={oos.get('sharpe',np.nan):.2f}"
              f" HitRate={oos.get('hit_rate',np.nan)*100:.0f}%"
              f" | BnH={oos.get('bnh',np.nan)*100:+.1f}%")
        rows.append({
            "crisis"      : name,
            "ticker"      : tkr,
            "name"        : stock_name,
            "market"      : market,
            "direction"   : direction,
            "total_cagr"  : tot["cagr"],
            "total_sharpe": tot["sharpe"],
            "oos_cagr"    : oos.get("cagr",  np.nan),
            "oos_sharpe"  : oos.get("sharpe", np.nan),
            "oos_mdd"     : oos.get("mdd",    np.nan),
            "oos_bnh"     : oos.get("bnh",    np.nan),
            "oos_hitrate" : oos.get("hit_rate", np.nan),
            "n_trade"     : tot["n_trade"],
        })

    return {
        "df"       : pd.DataFrame(rows),
        "beta"     : beta_30,
        "vrp"      : vrp,
        "signal"   : signal,
        "etf_price": prices[etf],
        "spy_price": prices["SPY"],
        "cfg"      : cfg,
    }


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------

def plot_results(results: dict):
    valid  = {k: v for k, v in results.items() if v is not None}
    n      = len(valid)
    keys   = list(valid.keys())

    fig = plt.figure(figsize=(22, 5 * n + 6))
    outer = gridspec.GridSpec(n + 1, 1, figure=fig, hspace=0.55)

    for row_idx, name in enumerate(keys):
        res  = valid[name]
        cfg  = res["cfg"]
        beta = res["beta"]
        vrp  = res["vrp"]
        sig  = res["signal"]
        ep   = res["etf_price"]
        sp   = res["spy_price"]
        df   = res["df"]
        ev   = pd.Timestamp(cfg["event_date"])
        oos  = pd.Timestamp(cfg["oos_start"])
        col  = cfg["color"]
        dirn = cfg["direction"]

        inner = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=outer[row_idx], wspace=0.38
        )

        # --- 패널 1: ETF vs SPY 누적 수익 ---
        ax0 = fig.add_subplot(inner[0])
        ci  = ep.index.intersection(sp.index)
        en  = ep.loc[ci] / ep.loc[ci].iloc[0] * 100
        sn  = sp.loc[ci] / sp.loc[ci].iloc[0] * 100
        ax0.plot(en, color=col, linewidth=1.4, label=cfg["etf"])
        ax0.plot(sn, color="gray", linewidth=0.9, alpha=0.7, label="SPY")
        ax0.axvline(ev,  color="red",  linestyle="--", linewidth=0.9, alpha=0.8,
                    label=cfg["event_label"].replace("\n", " "))
        ax0.axvline(oos, color="blue", linestyle=":",  linewidth=0.8, alpha=0.6,
                    label="OOS 시작")
        sig_dates = sig[sig == True].index.intersection(ci)
        for sd in sig_dates[::1]:
            ax0.axvspan(sd, sd + pd.Timedelta(days=2), alpha=0.07, color=col)
        ax0.set_title(f"{cfg['label']}\n{cfg['etf']} vs SPY (기준=100)", fontsize=8)
        ax0.legend(fontsize=6, loc="upper left")
        ax0.tick_params(labelsize=7)

        # --- 패널 2: Rolling beta + VRP ---
        ax1  = fig.add_subplot(inner[1])
        ax1b = ax1.twinx()
        b_p  = beta.reindex(ci).dropna()
        v_p  = vrp.reindex(ci).dropna()
        ax1.plot(b_p,  color="navy",       linewidth=1.1, label="Beta(30D)", alpha=0.85)
        ax1b.plot(v_p, color="darkorange", linewidth=0.8, label="VRP",       alpha=0.75)
        ax1.axhline(cfg["beta_thresh"], color="navy",       linestyle="--",
                    linewidth=0.7, alpha=0.55)
        ax1b.axhline(0,                color="darkorange", linestyle="--",
                     linewidth=0.7, alpha=0.55)
        ax1.axvline(ev,  color="red",  linestyle="--", linewidth=0.8, alpha=0.7)
        ax1.axvline(oos, color="blue", linestyle=":",  linewidth=0.7, alpha=0.6)
        thresh_str = (f"< {cfg['beta_thresh']}" if dirn == "LONG"
                      else f"> {cfg['beta_thresh']}")
        ax1.set_title(f"Beta(30D) & VRP\n신호 조건: beta {thresh_str} AND VRP<0", fontsize=8)
        ax1.set_ylabel("Beta", fontsize=7, color="navy")
        ax1b.set_ylabel("VRP(ann)", fontsize=7, color="darkorange")
        ax1.tick_params(labelsize=7)
        ax1b.tick_params(labelsize=7)
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax1b.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, fontsize=6)

        # --- 패널 3: OOS CAGR 바 차트 ---
        ax2  = fig.add_subplot(inner[2])
        df_v = df.dropna(subset=["oos_cagr"])
        if len(df_v) > 0:
            bar_colors = [col if v >= 0 else "#999999" for v in df_v["oos_cagr"]]
            ax2.bar(range(len(df_v)), df_v["oos_cagr"] * 100,
                    color=bar_colors, alpha=0.75)
            ax2.axhline(0, color="black", linewidth=0.5)
            ax2.set_xticks(range(len(df_v)))
            ax2.set_xticklabels(
                [t.split(".")[0] for t in df_v["ticker"]],
                rotation=35, fontsize=7
            )
            ax2.set_ylabel("OOS CAGR (%)", fontsize=8)
            # BnH overlay
            ax2b = ax2.twinx()
            ax2b.plot(range(len(df_v)), df_v["oos_bnh"] * 100,
                      marker="o", markersize=4, color="dimgray",
                      linewidth=0.8, label="BnH")
            ax2b.set_ylabel("BnH OOS (%)", fontsize=7, color="dimgray")
            ax2b.tick_params(labelsize=7)
            ax2b.legend(fontsize=6)
        ax2.set_title(f"OOS CAGR ({dirn} 전략)\nvs BnH (회색선)", fontsize=8)
        ax2.tick_params(labelsize=7)

    # --- 하단: 위기별 요약 테이블 ---
    all_dfs = pd.concat([v["df"] for v in valid.values()], ignore_index=True)

    summary_rows = []
    for crisis_name, res in valid.items():
        cfg_ = res["cfg"]
        d    = res["df"].dropna(subset=["oos_cagr"])
        if len(d) == 0:
            continue
        mean_cagr   = d["oos_cagr"].mean()
        mean_sharpe = d["oos_sharpe"].mean()
        mean_bnh    = d["oos_bnh"].mean()
        pct_pos     = (d["oos_cagr"] > 0).mean()
        mean_hit    = d["oos_hitrate"].mean()
        summary_rows.append([
            cfg_["label"],
            cfg_["direction"],
            f"{mean_cagr*100:+.1f}%",
            f"{mean_sharpe:.2f}",
            f"{mean_bnh*100:+.1f}%",
            f"{pct_pos*100:.0f}%",
            f"{mean_hit*100:.0f}%",
        ])

    ax_t = fig.add_subplot(outer[n])
    ax_t.axis("off")
    if summary_rows:
        col_labels = ["위기", "방향", "평균OOS CAGR", "평균Sharpe",
                      "평균BnH CAGR", "수익종목비율", "5D 히트레이트"]
        tbl = ax_t.table(
            cellText  = summary_rows,
            colLabels = col_labels,
            cellLoc   = "center",
            loc       = "center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9.5)
        tbl.scale(1.0, 2.2)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor("#2c3e50")
                cell.set_text_props(color="white", weight="bold")
            elif "SHORT" in str(cell.get_text().get_text()):
                cell.set_facecolor("#fde8e8")
            elif "LONG" in str(cell.get_text().get_text()):
                cell.set_facecolor("#e8fde8")
    ax_t.set_title("위기별 전략 성과 요약 (OOS 기간 기준)",
                   fontsize=12, pad=18, weight="bold")

    fig.suptitle(
        "역사적 위기 × 섹터 매핑 — 프레임워크 일반화 검증\n"
        "신호: 섹터 ETF Beta 체제전환 + VRP 음전환 (EWMA 기반)  |  "
        "LONG=수혜섹터, SHORT=피해섹터",
        fontsize=13, y=1.01,
    )

    out = FIGURES_DIR / "historical_crisis_test.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  차트 저장: {out}")


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sep = "=" * 66
    print(f"\n{sep}")
    print("  역사적 위기 프레임워크 일반화 검증")
    print(f"  신호: beta 체제전환 + VRP 음전환 (EWMA, consec={CONSEC}일)")
    print(f"{sep}")

    all_results = {}
    for crisis_key, cfg in CRISES.items():
        res = run_crisis(crisis_key, cfg)
        all_results[crisis_key] = res

    # CSV 저장
    valid_dfs = [v["df"] for v in all_results.values() if v is not None]
    if valid_dfs:
        all_df  = pd.concat(valid_dfs, ignore_index=True)
        out_csv = OUTPUT_DIR / "crisis_results.csv"
        all_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"\n  결과 CSV: {out_csv}")

    print("\n  [시각화 생성 중...]")
    plot_results(all_results)

    print(f"\n{sep}")
    print("  완료.")
    print(f"{sep}\n")

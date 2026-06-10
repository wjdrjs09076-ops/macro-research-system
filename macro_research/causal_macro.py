"""
causal_macro.py
PCMCI 기반 매크로 팩터 인과 발견 파이프라인.

Stage 1  데이터 수집 + 월별 정렬
Stage 2  정상성(ADF) + 다중공선성(VIF) 진단
Stage 3  Change-point detection → 현재 레짐 구간 탐지
Stage 4  PCMCI × 2 (전체 기간 + 현재 레짐)
Stage 5  링크 안정성 분류 (structural / emerging / historical)
         → output/causal_links.csv

Usage:
  python causal_macro.py
"""

import warnings; warnings.filterwarnings("ignore")
import os, sys, json, urllib.request, urllib.parse
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
from pathlib import Path

# ── PCMCI ─────────────────────────────────────────────────────────────────────
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

# ── Stats ──────────────────────────────────────────────────────────────────────
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
START    = "2010-01-01"
END      = date.today().isoformat()
TAU_MAX  = 6      # 최대 lag (월 단위)
PC_ALPHA = 0.05   # PCMCI 유의 수준
VIF_WARN = 10.0   # VIF 경고 임계값

FRED_KEY = os.environ.get("FRED_API_KEY", "")

FRED_MAP = {
    "CREDIT_SPREAD": "BAA10Y",   # Moody's Baa - 10Y Treasury 스프레드 (1986~, 무료)
    "CPI":           "CPIAUCSL", # 소비자물가 (월별)
    "FED_FUNDS":     "FEDFUNDS", # 기준금리 (월별)
}

# ── Stage 1: 데이터 수집 ───────────────────────────────────────────────────────

def fetch_yfinance_monthly() -> pd.DataFrame:
    """VIX, US10Y, US2Y, DXY, OIL → 월말 수준값."""
    tickers = {"VIX": "^VIX", "US10Y": "^TNX",
               "US2Y": "^IRX", "DXY": "DX-Y.NYB", "OIL": "CL=F"}
    raw = yf.download(list(tickers.values()), start=START, end=END,
                      auto_adjust=True, progress=False, timeout=30)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw

    frames = {}
    for name, sym in tickers.items():
        if sym in close.columns:
            s = close[sym].dropna()
            frames[name] = s.resample("ME").last()

    return pd.DataFrame(frames)


def fetch_fred_monthly(series_id: str) -> pd.Series:
    if not FRED_KEY:
        return pd.Series(dtype=float)
    url = (f"https://api.stlouisfed.org/fred/series/observations"
           f"?series_id={series_id}&api_key={FRED_KEY}&file_type=json"
           f"&observation_start={START}&sort_order=asc&limit=100000")
    req = urllib.request.Request(url, headers={"User-Agent": "MacroMonitor/1.0"})
    with urllib.request.urlopen(req, timeout=12) as r:
        obs = json.loads(r.read().decode()).get("observations", [])
    vals = {}
    for o in obs:
        v = o.get("value", ".")
        if v != ".":
            try:
                vals[o["date"]] = float(v)
            except ValueError:
                pass
    s = pd.Series(vals)
    s.index = pd.to_datetime(s.index)
    return s.resample("ME").last().sort_index()


def fetch_china_pmi_monthly() -> pd.Series:
    try:
        import akshare as ak
        df = ak.macro_china_pmi_ylhz()
        # akshare PMI: 날짜 컬럼명이 다를 수 있음
        date_col  = df.columns[0]
        value_col = df.columns[1]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        s = pd.Series(df[value_col].values, index=df[date_col])
        s = pd.to_numeric(s, errors="coerce").dropna()
        return s.resample("ME").last().sort_index()
    except Exception as e:
        print(f"  [akshare PMI] {e}")
        return pd.Series(dtype=float)


def build_dataset() -> pd.DataFrame:
    print("[Stage 1] 데이터 수집 중...")

    yf_df = fetch_yfinance_monthly()
    print(f"  yfinance: {list(yf_df.columns)}  ({len(yf_df)} 월)")

    import time
    fred_frames = {}
    if FRED_KEY:
        for name, sid in FRED_MAP.items():
            try:
                s = fetch_fred_monthly(sid)
                if len(s) > 0:
                    fred_frames[name] = s
                    print(f"  FRED {name}: {len(s)} 월")
            except Exception as e:
                print(f"  FRED {name}: {e}")
            time.sleep(1.0)
    else:
        print("  FRED_API_KEY 없음 — FRED 시리즈 스킵")

    pmi = fetch_china_pmi_monthly()
    if len(pmi) > 10:
        fred_frames["CHINA_PMI"] = pmi
        print(f"  akshare China PMI: {len(pmi)} 월")

    df = yf_df.copy()
    for name, s in fred_frames.items():
        df[name] = s

    df = df.dropna(how="all")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


# ── Stage 2a: 정상성 (ADF) ────────────────────────────────────────────────────

def make_stationary(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    각 시계열 ADF 검정 → 단위근 있으면 1차 차분.
    반환: (정상화된 df, 결과 dict)
    """
    print("\n[Stage 2a] ADF 정상성 검정...")
    results = {}
    out = {}

    for col in df.columns:
        s = df[col].dropna()
        if len(s) < 20:
            continue
        adf_stat, p_val, _, _, crit, _ = adfuller(s, autolag="AIC")
        is_stationary = p_val < 0.05

        if is_stationary:
            out[col] = s
            transformed = "level"
        else:
            # 1차 차분
            diff = s.diff().dropna()
            adf2, p2, _, _, _, _ = adfuller(diff, autolag="AIC")
            out[col] = diff
            transformed = "diff"
            p_val = p2

        results[col] = {"adf": round(adf_stat, 3), "p": round(p_val, 4),
                        "transform": transformed, "stationary": is_stationary or transformed == "diff"}
        status = "OK" if results[col]["stationary"] else "WARN"
        print(f"  [{status}] {col:<14} p={p_val:.4f}  transform={transformed}")

    return pd.DataFrame(out).dropna(), results


# ── Stage 2b: 다중공선성 (VIF) ────────────────────────────────────────────────

def check_vif(df: pd.DataFrame) -> pd.DataFrame:
    """VIF 계산 + 경고 플래그."""
    print("\n[Stage 2b] VIF 다중공선성 진단...")
    clean = df.dropna()
    vif_data = []
    X = clean.values

    for i, col in enumerate(clean.columns):
        try:
            v = variance_inflation_factor(X, i)
        except Exception:
            v = float("nan")
        flag = " ← HIGH" if v > VIF_WARN else ""
        vif_data.append({"variable": col, "VIF": round(v, 2), "flag": flag})
        print(f"  {col:<14} VIF={v:.2f}{flag}")

    print("\n  상관계수 행렬 (|r| > 0.7 주목):")
    corr = clean.corr().round(2)
    for i, c1 in enumerate(corr.columns):
        for c2 in corr.columns[i+1:]:
            r = corr.loc[c1, c2]
            if abs(r) > 0.70:
                print(f"  {c1} <-> {c2}: r={r:+.2f}  ← 높은 공선성")

    return pd.DataFrame(vif_data)


# ── Stage 3: Change-point detection ──────────────────────────────────────────

def detect_changepoints(df: pd.DataFrame, window: int = 24, top_n: int = 4,
                        min_gap: int = 12) -> list:
    """
    롤링 공분산 행렬 Frobenius 거리로 구조 변환점 탐지.
    window  : 비교 윈도우 크기 (개월)
    top_n   : 반환할 변환점 최대 수
    min_gap : 변환점 간 최소 간격 (개월)
    반환: [(date, score), ...]  시간순
    """
    from sklearn.preprocessing import StandardScaler
    from scipy.signal import find_peaks

    clean = df.dropna()
    n = len(clean)
    if n < window * 2 + min_gap:
        print(f"  [WARN] 데이터 부족 ({n}개월) -- change-point 스킵")
        return []

    scaled = StandardScaler().fit_transform(clean.values)

    # 인접 두 윈도우의 공분산 행렬 Frobenius 거리
    distances, dates = [], []
    for i in range(window, n - window):
        w1 = scaled[i - window:i]
        w2 = scaled[i:i + window]
        cov1 = np.cov(w1.T)
        cov2 = np.cov(w2.T)
        dist = float(np.linalg.norm(cov1 - cov2, "fro"))
        distances.append(dist)
        dates.append(clean.index[i])

    dist_arr = np.array(distances)

    # 피크 탐지 (prominence 기준: 표준편차 0.5배 이상)
    peaks, _ = find_peaks(dist_arr,
                          distance=min_gap,
                          prominence=dist_arr.std() * 0.5)

    # top_n개만 유지
    if len(peaks) > top_n:
        top_idx = np.argsort(dist_arr[peaks])[-top_n:]
        peaks = np.sort(peaks[top_idx])

    result = [(dates[p], round(dist_arr[p], 4)) for p in peaks]

    print(f"\n[Stage 3] Change-point 탐지 (window={window}개월)...")
    for dt, sc in result:
        print(f"  {dt.strftime('%Y-%m')}  score={sc:.3f}")
    return result


# ── Stage 4: PCMCI ────────────────────────────────────────────────────────────

def run_pcmci(df: pd.DataFrame, label: str = "") -> dict:
    """
    PCMCI (ParCorr) 실행.
    tau_max를 데이터 길이에 맞게 자동 조정.
    반환: {'val_matrix', 'p_matrix', 'var_names', 'tau_max'}
    """
    clean = df.dropna()
    if len(clean) < 24:
        raise ValueError(f"데이터 부족: {len(clean)}행 (최소 24)")

    # 관측 수에 따라 tau_max 자동 조정
    effective_tau = min(TAU_MAX, max(2, len(clean) // 8))
    tag = f"[{label}] " if label else ""
    print(f"\n[Stage 4] {tag}PCMCI 실행  "
          f"(obs={len(clean)}, tau_max={effective_tau}, alpha={PC_ALPHA})...")

    data_arr  = clean.values.astype(float)
    var_names = list(clean.columns)

    dataframe = pp.DataFrame(data_arr, var_names=var_names)
    pcmci     = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(significance="analytic"),
                      verbosity=0)
    results   = pcmci.run_pcmci(tau_min=1, tau_max=effective_tau, pc_alpha=PC_ALPHA)

    print(f"  완료. 변수 {len(var_names)}개  관측 {len(clean)}개월")
    return {
        "val_matrix": results["val_matrix"],
        "p_matrix":   results["p_matrix"],
        "var_names":  var_names,
        "tau_max":    effective_tau,
        "n_obs":      len(clean),
    }


# ── Stage 5: 링크 추출 + 안정성 분류 ─────────────────────────────────────────

def extract_links(pcmci_result: dict, alpha: float = PC_ALPHA) -> pd.DataFrame:
    """p_matrix에서 유의미한 인과 링크만 추출."""
    val = pcmci_result["val_matrix"]
    p   = pcmci_result["p_matrix"]
    nms = pcmci_result["var_names"]

    rows = []
    for j, target in enumerate(nms):
        for i, source in enumerate(nms):
            if i == j:
                continue
            for lag in range(1, pcmci_result["tau_max"] + 1):
                pv = p[i, j, lag]
                coef = val[i, j, lag]
                if pv < alpha:
                    rows.append({
                        "source":    source,
                        "target":    target,
                        "lag_months": lag,
                        "coef":      round(float(coef), 4),
                        "p_value":   round(float(pv), 5),
                        "direction": "+" if coef > 0 else "-",
                    })

    df = pd.DataFrame(rows).sort_values("p_value")
    return df


def classify_stability(full: pd.DataFrame, recent: pd.DataFrame) -> pd.DataFrame:
    """
    전체 기간 링크 vs 현재 레짐 링크를 비교해 안정성 분류.

    structural : 양쪽 모두 유의  → 구조적 인과, 신뢰도 최고
    emerging   : 최근만 유의     → 현재 레짐 신호, 최신 반영
    historical : 전체만 유의     → 과거 레짐 산물, 가중치 하향

    stability_score:
      structural  = 0.65*(1-p_recent) + 0.35*(1-p_full)
      emerging    = 0.85*(1-p_recent)
      historical  = 0.20*(1-p_full)
    """
    def key(r):
        return (r["source"], r["target"], r["lag_months"], r["direction"])

    full_dict   = {key(r): r for _, r in full.iterrows()}
    recent_dict = {key(r): r for _, r in recent.iterrows()}
    all_keys    = set(full_dict) | set(recent_dict)

    rows = []
    for k in all_keys:
        in_full   = k in full_dict
        in_recent = k in recent_dict
        base = full_dict[k] if in_full else recent_dict[k]

        p_full   = full_dict[k]["p_value"]   if in_full   else 1.0
        p_recent = recent_dict[k]["p_value"] if in_recent else 1.0
        c_full   = full_dict[k]["coef"]      if in_full   else float("nan")
        c_recent = recent_dict[k]["coef"]    if in_recent else float("nan")

        if in_full and in_recent:
            label = "structural"
            score = 0.65*(1-p_recent) + 0.35*(1-p_full)
        elif in_recent:
            label = "emerging"
            score = 0.85*(1-p_recent)
        else:
            label = "historical"
            score = 0.20*(1-p_full)

        rows.append({
            "source":      base["source"],
            "target":      base["target"],
            "lag_months":  base["lag_months"],
            "direction":   base["direction"],
            "coef_full":   round(c_full,   4),
            "coef_recent": round(c_recent, 4),
            "p_full":      round(p_full,   5),
            "p_recent":    round(p_recent, 5),
            "stability":   label,
            "score":       round(score,    4),
        })

    df = pd.DataFrame(rows).sort_values(["stability", "score"],
                                         ascending=[True, False])
    return df


def print_stability_report(df: pd.DataFrame) -> None:
    LABELS = ["structural", "emerging", "historical"]
    ICONS  = {"structural": "**", "emerging": ">>", "historical": ".."}

    print(f"\n{'='*78}")
    print(f"  링크 안정성 분류  (전체 {len(df)}개)")
    print(f"  ** structural = 양쪽 유의  |  >> emerging = 최근만 유의  "
          f"|  .. historical = 과거만 유의")
    print(f"{'='*78}")
    print(f"  {'':2} {'Source':<13} {'Target':<13} {'Lag':>4}  "
          f"{'Coef(F)':>8} {'Coef(R)':>8}  {'p(F)':>7} {'p(R)':>7}  {'Score':>6}")
    print(f"  {'-'*72}")

    for label in LABELS:
        sub = df[df["stability"] == label]
        if len(sub) == 0:
            continue
        for _, r in sub.iterrows():
            icon  = ICONS[label]
            arrow = "-->" if r["direction"] == "+" else "--X"
            cf    = f"{r['coef_full']:+.3f}"   if not pd.isna(r['coef_full'])   else "  n/a "
            cr    = f"{r['coef_recent']:+.3f}" if not pd.isna(r['coef_recent']) else "  n/a "
            pf    = f"{r['p_full']:.4f}"
            pr    = f"{r['p_recent']:.4f}"
            print(f"  {icon} {r['source']:<13} {r['target']:<13} "
                  f"{r['lag_months']:>3}m  {cf:>8} {cr:>8}  {pf:>7} {pr:>7}  "
                  f"{r['score']:>6.3f}  {arrow}")
        print()

    print(f"{'='*78}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    # Stage 1
    raw_df = build_dataset()
    print(f"\n  수집 완료: {raw_df.shape[0]}개월 × {raw_df.shape[1]}개 변수")
    print(f"  기간: {raw_df.index[0].date()} ~ {raw_df.index[-1].date()}")
    print(f"  변수: {list(raw_df.columns)}")

    # Stage 2a + 2b
    stat_df, adf_results = make_stationary(raw_df)
    vif_df = check_vif(stat_df)
    vif_df.to_csv(OUTPUT_DIR / "causal_vif.csv", index=False)

    # Stage 3: change-point detection
    MIN_RECENT = 30  # PCMCI에 최소 필요한 관측수
    changepoints = detect_changepoints(stat_df, window=24, top_n=4)

    # 가장 최근 changepoint부터 역순으로 탐색 — 구간이 너무 짧으면 이전 changepoint 사용
    last_cp = None
    for cp_date, _ in reversed(changepoints):
        candidate = stat_df[stat_df.index >= cp_date]
        if len(candidate) >= MIN_RECENT:
            last_cp = cp_date
            break
    if last_cp is None:
        # 모든 changepoint가 너무 최근 → 전체 데이터의 최근 MIN_RECENT 개월 사용
        last_cp = stat_df.index[-MIN_RECENT]
        print(f"  [WARN] 유효한 changepoint 없음 -- 최근 {MIN_RECENT}개월 사용")

    recent_df = stat_df[stat_df.index >= last_cp]
    print(f"\n  현재 레짐 구간: {last_cp.strftime('%Y-%m')} ~ "
          f"{stat_df.index[-1].strftime('%Y-%m')}  ({len(recent_df)}개월)")

    # Stage 4: PCMCI × 2
    full_result   = run_pcmci(stat_df,   label="전체 기간")
    recent_result = run_pcmci(recent_df, label="현재 레짐")

    # Stage 5: 링크 추출 + 안정성 분류
    full_links   = extract_links(full_result)
    recent_links = extract_links(recent_result)
    stability_df = classify_stability(full_links, recent_links)
    print_stability_report(stability_df)

    # 저장
    stability_df.to_csv(OUTPUT_DIR / "causal_links.csv", index=False)
    print(f"  저장 완료 -> output/causal_links.csv ({len(stability_df)}개 링크)")

    # Change-point 저장
    if changepoints:
        cp_df = pd.DataFrame(changepoints, columns=["date", "score"])
        cp_df.to_csv(OUTPUT_DIR / "causal_changepoints.csv", index=False)
        print(f"  저장 완료 -> output/causal_changepoints.csv")

    # p/val matrix 저장 (히트맵용)
    var_names = full_result["var_names"]
    for tau in range(1, full_result["tau_max"] + 1):
        pd.DataFrame(full_result["p_matrix"][:, :, tau],
                     index=var_names, columns=var_names
                     ).to_csv(OUTPUT_DIR / f"causal_p_lag{tau}.csv")
        pd.DataFrame(full_result["val_matrix"][:, :, tau],
                     index=var_names, columns=var_names
                     ).to_csv(OUTPUT_DIR / f"causal_val_lag{tau}.csv")

    print(f"  lag별 p/val 행렬 -> output/causal_p_lag*.csv")
    return stability_df, full_result, recent_result


if __name__ == "__main__":
    links, full_res, recent_res = run()

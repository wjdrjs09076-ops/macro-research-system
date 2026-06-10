"""
vol_monitor_pipeline.py
Scheduled pipeline version of research_vol_monitor.py.
Outputs vol_monitor_data.json to macro-portal/public/data/
Run via Windows Task Scheduler daily.
"""

import warnings; warnings.filterwarnings("ignore")
import os, sys, json, time, urllib.request, urllib.parse
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, timedelta, datetime

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__),
    "macro-portal", "public", "data", "vol_monitor_data.json"
)

ETF_UNIVERSE = [
    # US Sectors
    ("XLE",  "Energy / Oil"),
    ("XLF",  "Financials / Banking"),
    ("SOXX", "Semiconductors"),
    ("XTN",  "Transportation / Shipping"),
    ("XLK",  "Technology"),
    ("XLV",  "Healthcare"),
    ("XAR",  "Defense / Aerospace"),
    ("XLB",  "Materials"),
    ("XLI",  "Industrials"),
    ("XLC",  "Communication"),
    ("XLY",  "Consumer Discretionary"),
    ("GLD",  "Gold"),
    ("TLT",  "Long-term Treasury"),
    # Futures — trade ~24h on CME Globex
    ("CL=F", "WTI Crude Futures"),
    ("ES=F", "S&P 500 Futures"),
    ("GC=F", "Gold Futures"),
    ("ZN=F", "10Y Treasury Futures"),
    # International ETFs (USD-denominated, US-listed)
    ("EWY",  "Korea (MSCI)"),
    ("EWJ",  "Japan (MSCI)"),
    ("FXI",  "China Large-Cap"),
    ("MCHI", "China Broad (MSCI)"),
    ("EWU",  "UK (MSCI)"),
    ("EEM",  "Emerging Markets"),
    # Global Indices (via yfinance)
    ("^KS11",     "KOSPI"),
    ("^HSI",      "Hang Seng"),
    ("^FTSE",     "FTSE 100"),
    ("^FTMC",     "FTSE 250 (UK 내수)"),
    ("^N225",     "Nikkei 225"),
    ("^STOXX50E", "Euro Stoxx 50"),
    ("000001.SS",  "Shanghai Composite"),
]

LOOKBACK_DAYS = 252
RECENT_DAYS   = 10
ALERT_Z       = 1.8
VRP_ALERT     = -0.05
BETA_BREAK    = 0.30
VOL_CONSEC    = 5

OIL_LOOKBACK  = 252
OIL_RECENT    = 10
OIL_Z_ALERT   = 2.0
INV_ALERT_PCT = 3.0

FRED_LOOKBACK  = 252
FRED_RECENT    = 5
FRED_Z_ALERT   = 1.8

CHINA_LOOKBACK = 252
CHINA_RECENT   = 5
CHINA_Z_ALERT  = 1.8
CHINA_INDICES = {
    "sh000300": ("CSI300",  "CSI 300 (중국 대형주)",   ["FXI", "MCHI"]),
    "sh000905": ("CSI500",  "CSI 500 (중국 중형주)",   ["MCHI"]),
    "sz399006": ("ChiNext", "ChiNext (중국 성장/기술)", ["MCHI"]),
    "sh000001": ("Shanghai","Shanghai Composite",      ["FXI", "MCHI"]),
}
FRED_SERIES = {
    "USEPUINDXD":   ("EPU",    "Econ Policy Uncertainty",  ["ALL"],         False),
    "BAMLH0A0HYM2": ("HY Spd", "HY Credit Spread",         ["XLF"],         True),
    "T10Y2Y":       ("Yld Crv","10Y-2Y Yield Curve",        ["XLF", "TLT"], False),
    "GEPUCURRENT":  ("GPR",    "Geopolitical Risk Index",   ["XAR", "XLE"], False),
}

END   = date.today().isoformat()
START = (date.today() - timedelta(days=LOOKBACK_DAYS + 60)).isoformat()
EIA_START  = (date.today() - timedelta(days=OIL_LOOKBACK  + 30)).isoformat()
FRED_START = (date.today() - timedelta(days=FRED_LOOKBACK + 90)).isoformat()


# ── Fetch helpers ─────────────────────────────────────────────────────────────

def fetch_etf_data():
    tickers = [t for t, _ in ETF_UNIVERSE] + ["SPY", "^VIX"]
    raw = yf.download(tickers, start=START, end=END,
                      auto_adjust=True, progress=False, timeout=30)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    out = {}
    for t in tickers:
        if t in close.columns:
            s = close[t].dropna()
            if len(s) > 30:
                out[t] = s
    return out


def eia_get(path, params, api_key):
    params["api_key"] = api_key
    params["length"]  = params.get("length", 500)
    qs  = urllib.parse.urlencode(params, doseq=True)
    url = f"https://api.eia.gov/v2/{path}?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": "MacroMonitor/1.0"})
    with urllib.request.urlopen(req, timeout=12) as r:
        return json.loads(r.read().decode("utf-8")).get("response", {}).get("data", [])


def fetch_eia(api_key):
    result = {}
    for series, key in [("RWTC", "wti"), ("RBRTE", "brent")]:
        try:
            rows = eia_get("petroleum/pri/spt/data/", {
                "frequency": "daily", "data[0]": "value",
                "facets[series][]": series, "start": EIA_START,
                "sort[0][column]": "period", "sort[0][direction]": "asc",
            }, api_key)
            if rows:
                s = pd.Series({r["period"]: float(r["value"])
                               for r in rows if r["value"] not in (None, "")})
                s.index = pd.to_datetime(s.index)
                result[key] = s.sort_index()
        except Exception as e:
            print(f"  [EIA] {series}: {e}")
    try:
        rows = eia_get("petroleum/stoc/wstk/data/", {
            "frequency": "weekly", "data[0]": "value",
            "facets[series][]": "WCRSTUS1", "start": EIA_START,
            "sort[0][column]": "period", "sort[0][direction]": "asc",
        }, api_key)
        if rows:
            s = pd.Series({r["period"]: float(r["value"])
                           for r in rows if r["value"] not in (None, "")})
            s.index = pd.to_datetime(s.index)
            result["inv"] = s.sort_index()
    except Exception as e:
        print(f"  [EIA] inventory: {e}")
    return result


def fetch_akshare_china():
    try:
        import akshare as ak
    except ImportError:
        print("  [akshare] not installed — skipping China data")
        return {}
    result = {}
    china_start = (date.today() - timedelta(days=CHINA_LOOKBACK + 60)).strftime("%Y%m%d")
    china_end   = date.today().strftime("%Y%m%d")
    for sym, (label, desc, etfs) in CHINA_INDICES.items():
        try:
            df = ak.stock_zh_index_daily(symbol=sym)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            s = df["close"].loc[china_start:china_end].dropna()
            if len(s) >= 30:
                result[sym] = s
        except Exception as e:
            print(f"  [akshare] {sym}: {e}")
    return result


def fetch_fred_series(sid, api_key):
    url = (f"https://api.stlouisfed.org/fred/series/observations"
           f"?series_id={sid}&api_key={api_key}&file_type=json"
           f"&observation_start={FRED_START}&sort_order=asc")
    req = urllib.request.Request(url, headers={"User-Agent": "MacroMonitor/1.0"})
    with urllib.request.urlopen(req, timeout=12) as r:
        obs = json.loads(r.read().decode("utf-8")).get("observations", [])
    vals = {}
    for o in obs:
        v = o.get("value", ".")
        if v != ".":
            try: vals[o["date"]] = float(v)
            except ValueError: pass
    s = pd.Series(vals)
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


# ── Compute functions ─────────────────────────────────────────────────────────

def compute_etf_signal(ticker, data):
    if ticker not in data or "SPY" not in data:
        return None
    rets     = data[ticker].pct_change().dropna()
    spy_rets = data["SPY"].pct_change().dropna()
    vix      = data.get("^VIX")
    if len(rets) < LOOKBACK_DAYS // 2:
        return None

    rv_series  = rets.rolling(RECENT_DAYS).std() * np.sqrt(252)
    current_rv = float(rv_series.iloc[-1])
    hist_rv    = rv_series.iloc[-LOOKBACK_DAYS:-RECENT_DAYS]
    baseline   = float(hist_rv.mean())
    std_rv     = float(hist_rv.std()) if hist_rv.std() > 0 else 1e-6
    z_score    = (current_rv - baseline) / std_rv

    idx = rets.index.intersection(spy_rets.index)
    s, m = rets.loc[idx], spy_rets.loc[idx]
    b63  = (s.rolling(63).cov(m) / m.rolling(63).var()).dropna().reindex(idx).ffill()
    res  = s - b63 * m
    idio = res.ewm(span=32, min_periods=20).std() * np.sqrt(252)
    vix_a = vix.reindex(idx).ffill() / 100.0 if vix is not None \
            else s.rolling(21).std() * np.sqrt(252) * 1.2
    capm_iv = np.sqrt((b63 * vix_a) ** 2 + idio ** 2)
    vrp_now = float((capm_iv - rv_series.reindex(idx)).iloc[-1])
    beta_30 = float((s.rolling(30).cov(m) / m.rolling(30).var()).iloc[-1])
    consec_vol = int((rv_series > float(hist_rv.median())).astype(int).iloc[-VOL_CONSEC:].sum())
    aligned    = rets.reindex(spy_rets.index).dropna()
    excess_10d = float((aligned.tail(10) - spy_rets.reindex(aligned.index).tail(10)).sum())

    reasons, level = [], 0
    if z_score >= ALERT_Z:
        reasons.append(f"vol z={z_score:.1f}"); level += 1
    if z_score >= ALERT_Z + 0.7: level += 1
    if vrp_now < VRP_ALERT:
        reasons.append(f"VRP={vrp_now:.3f}"); level += 1
    if abs(beta_30 - 1.0) > BETA_BREAK:
        reasons.append(f"beta={beta_30:.2f}"); level = max(level, 1)
    if consec_vol >= VOL_CONSEC:
        reasons.append(f"{consec_vol}d consec vol"); level = max(level, 1)

    label = ["---", "WATCH", "ALERT", "HIGH"][min(level, 3)]
    return dict(ticker=ticker, current_rv=round(current_rv*100,1),
                baseline_rv=round(baseline*100,1), z_score=round(z_score,2),
                vrp=round(vrp_now,4), beta_30d=round(beta_30,3),
                consec_vol=consec_vol, excess_10d=round(excess_10d*100,2),
                level=level, label=label, reason=", ".join(reasons) if reasons else "")


def compute_oil(eia_raw):
    out = {}
    for key, label in [("wti","WTI"), ("brent","Brent")]:
        s = eia_raw.get(key)
        if s is None or len(s) < 60: continue
        recent = float(s.tail(OIL_RECENT).mean())
        base   = s.iloc[-OIL_LOOKBACK:-OIL_RECENT]
        bm, bs = float(base.mean()), float(base.std()) if base.std() > 0 else 1e-6
        z      = (recent - bm) / bs
        ma60   = float(s.rolling(60).mean().iloc[-1])
        out[key] = dict(label=label, current=round(recent,2), baseline=round(bm,2),
                        z_score=round(z,2), bw_spread=round((recent/ma60-1)*100,2))
    inv = eia_raw.get("inv")
    if inv is not None and len(inv) >= 4:
        latest, prev = float(inv.iloc[-1]), float(inv.iloc[-2])
        chg  = (latest/prev - 1) * 100
        avg  = float(inv.pct_change().dropna().tail(12).mean() * 100)
        out["inv"] = dict(latest_mbbl=round(latest,1), wow_chg_pct=round(chg,2),
                          surprise=round(chg-avg,2))
    level, reasons = 0, []
    for k in ("wti","brent"):
        d = out.get(k)
        if d and abs(d["z_score"]) >= OIL_Z_ALERT:
            reasons.append(f"{d['label']} z={d['z_score']:+.1f}"); level += 1
        if d and d["bw_spread"] >= 5.0:
            reasons.append(f"{d['label']} backwardation +{d['bw_spread']:.1f}%")
            level = max(level, 1)
    inv = out.get("inv")
    if inv and abs(inv["surprise"]) >= INV_ALERT_PCT:
        reasons.append(f"inventory surprise {inv['surprise']:+.1f}%"); level = max(level, 1)
    out["level"]   = min(level, 3)
    out["reasons"] = reasons
    return out


def compute_fred(fred_raw):
    out = []
    for sid, s in fred_raw.items():
        label, desc, etfs, invert = FRED_SERIES[sid]
        s_valid = s.dropna()
        if len(s_valid) < 30: continue
        recent  = float(s_valid.tail(FRED_RECENT).mean())
        base    = s_valid.iloc[-(FRED_LOOKBACK+FRED_RECENT):-FRED_RECENT]
        if len(base) < 20: base = s_valid.iloc[:-FRED_RECENT]
        bm, bs  = float(base.mean()), float(base.std()) if base.std() > 0 else 1e-6
        z       = (recent - bm) / bs
        if invert: z = -z
        out.append(dict(sid=sid, label=label, desc=desc, etfs=etfs,
                        current=round(recent,4), baseline=round(bm,4),
                        z_score=round(z,2), last_date=s_valid.index[-1].strftime("%Y-%m-%d")))
    return out


def compute_china(china_raw):
    out = []
    for sym, s in china_raw.items():
        label, desc, etfs = CHINA_INDICES[sym]
        s_valid = s.dropna()
        if len(s_valid) < 30:
            continue
        recent = float(s_valid.tail(CHINA_RECENT).mean())
        base   = s_valid.iloc[-(CHINA_LOOKBACK + CHINA_RECENT):-CHINA_RECENT]
        if len(base) < 20:
            base = s_valid.iloc[:-CHINA_RECENT]
        bm, bs = float(base.mean()), float(base.std()) if base.std() > 0 else 1e-6
        z      = (recent - bm) / bs
        rets   = s_valid.pct_change().dropna()
        rv_20  = float(rets.tail(20).std() * (252 ** 0.5)) if len(rets) >= 20 else None
        out.append(dict(
            sym       = sym,
            label     = label,
            desc      = desc,
            etfs      = etfs,
            current   = round(recent, 2),
            baseline  = round(bm, 2),
            z_score   = round(z, 2),
            rv_20d    = round(rv_20 * 100, 2) if rv_20 is not None else None,
            last_date = s_valid.index[-1].strftime("%Y-%m-%d"),
        ))
    return out


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetching ETF data...")
    etf_data = fetch_etf_data()

    etf_signals = []
    for ticker, sector in ETF_UNIVERSE:
        r = compute_etf_signal(ticker, etf_data)
        if r:
            r["sector"] = sector
            etf_signals.append(r)
    etf_signals.sort(key=lambda x: -x["z_score"])

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetching EIA oil data...")
    oil_signals = {}
    eia_key = os.environ.get("EIA_API_KEY", "")
    if eia_key:
        try:
            oil_signals = compute_oil(fetch_eia(eia_key))
        except Exception as e:
            print(f"  [EIA] {e}")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetching China index data (akshare)...")
    china_signals = []
    try:
        china_raw     = fetch_akshare_china()
        china_signals = compute_china(china_raw)
    except Exception as e:
        print(f"  [akshare] {e}")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetching FRED signals...")
    fred_signals = []
    fred_key = os.environ.get("FRED_API_KEY", "")
    if fred_key:
        fred_raw = {}
        for sid in FRED_SERIES:
            try:
                s = fetch_fred_series(sid, fred_key)
                if len(s) >= 20:
                    fred_raw[sid] = s
            except Exception as e:
                print(f"  [FRED] {sid}: {e}")
            time.sleep(0.6)
        fred_signals = compute_fred(fred_raw)

    flagged_count = sum(1 for r in etf_signals if r["level"] >= 1)
    alert_count   = sum(1 for r in etf_signals if r["level"] >= 2)
    high_count    = sum(1 for r in etf_signals if r["level"] >= 3)

    payload = dict(
        scan_date     = datetime.now().isoformat(timespec="seconds"),
        etf_signals   = etf_signals,
        oil           = oil_signals,
        fred          = fred_signals,
        china         = china_signals,
        flagged_count = flagged_count,
        alert_count   = alert_count,
        high_count    = high_count,
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved -> {OUTPUT_PATH}")
    return payload


if __name__ == "__main__":
    run()

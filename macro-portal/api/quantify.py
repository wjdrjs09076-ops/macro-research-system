"""
quantify.py — Vercel Python serverless function
Gate Score + Simplified EVT (MEF/Hill/Stability/GPD) 분석
yfinance 데이터 → 실시간 계산
"""

from http.server import BaseHTTPRequestHandler
import json
import traceback

EWMA_SPAN   = 32
RV_WIN      = 20
BETA_WIN    = 30
GATE_WINDOW = 15
GATE_CONSEC = 5
GATE_THRESH = 0.45


# ─── Data fetch ──────────────────────────────────────────────────────────────

def fetch(ticker: str, start: str, end: str) -> dict:
    import yfinance as yf
    import pandas as pd

    syms = list({ticker, "SPY", "^VIX"})
    raw  = yf.download(syms, start=start, end=end,
                       auto_adjust=True, progress=False, timeout=20)
    close = raw["Close"] if isinstance(raw.columns, __import__("pandas").MultiIndex) else raw
    out = {}
    for t in syms:
        if t in close.columns:
            s = close[t].dropna()
            if len(s) > 20:
                out[t] = s
    return out


# ─── Gate computation ─────────────────────────────────────────────────────────

def compute_gate(data: dict, ticker: str, direction: str):
    import numpy as np
    import pandas as pd

    if ticker not in data or "SPY" not in data:
        raise ValueError(f"Insufficient data for {ticker}")

    sec_ret = data[ticker].pct_change().dropna()
    spy_ret = data["SPY"].pct_change().dropna()
    vix     = data.get("^VIX")

    idx = sec_ret.index.intersection(spy_ret.index)
    s, m = sec_ret.loc[idx], spy_ret.loc[idx]

    # Rolling beta (30d)
    beta = (s.rolling(BETA_WIN).cov(m) / m.rolling(BETA_WIN).var()).dropna()

    # VRP
    b63    = (s.rolling(63).cov(m) / m.rolling(63).var()).dropna().reindex(idx).ffill()
    res    = s - b63 * m
    idio   = res.ewm(span=EWMA_SPAN, min_periods=20).std() * np.sqrt(252)
    if vix is not None:
        vix_a = vix.reindex(idx).ffill() / 100.0
    else:
        vix_a = s.rolling(21).std() * np.sqrt(252) * 1.2
    capm_iv = np.sqrt((b63 * vix_a) ** 2 + idio ** 2)
    rv_ann  = s.rolling(RV_WIN).std() * np.sqrt(252)
    vrp     = (capm_iv - rv_ann).dropna()

    idx2 = beta.index.intersection(vrp.index).intersection(idx)
    b    = beta.reindex(idx2)
    v    = vrp.reindex(idx2)
    excs = s.reindex(idx2) - m.reindex(idx2)

    if direction == "LONG":
        beta_comp = (b < -0.05).astype(float).rolling(GATE_WINDOW, min_periods=5).mean()
        dir_comp  = (excs > 0).astype(float).rolling(GATE_WINDOW, min_periods=5).mean()
    else:
        beta_comp = (b > 1.50).astype(float).rolling(GATE_WINDOW, min_periods=5).mean()
        dir_comp  = (excs < 0).astype(float).rolling(GATE_WINDOW, min_periods=5).mean()

    vrp_comp   = (v < 0).astype(float).rolling(GATE_WINDOW, min_periods=5).mean()
    gate_score = 0.40 * beta_comp + 0.35 * vrp_comp + 0.25 * dir_comp

    above     = (gate_score >= GATE_THRESH).astype(float)
    gate_open = above.rolling(GATE_CONSEC, min_periods=GATE_CONSEC).min().fillna(0).astype(bool)

    return pd.DataFrame({
        "beta_comp":  beta_comp,
        "vrp_comp":   vrp_comp,
        "dir_comp":   dir_comp,
        "gate_score": gate_score,
        "gate_open":  gate_open,
    }, index=idx2).dropna(subset=["gate_score"])


# ─── EVT (no GARCH; raw return losses) ───────────────────────────────────────

def compute_evt(returns) -> dict:
    import numpy as np
    import pandas as pd
    from scipy.stats import genpareto

    losses = (-returns).dropna()
    losses = losses[losses > 0].values
    if len(losses) < 30:
        return {}

    n           = len(losses)
    losses_desc = np.sort(losses)[::-1]

    # MEF
    u_pcts = np.linspace(55, 92, 45)
    mef = []
    for p in u_pcts:
        u   = np.percentile(losses, p)
        exc = losses[losses > u] - u
        if len(exc) >= 8:
            mef.append({"u": round(float(u), 5),
                        "mef": round(float(exc.mean()), 5),
                        "n": int(len(exc))})

    # Hill estimator
    k_max = min(n // 2, 120)
    hill = []
    for k in range(8, k_max):
        if k < len(losses_desc) and losses_desc[k] > 0:
            h = float(np.mean(np.log(losses_desc[:k] / losses_desc[k])))
            hill.append({"k": k,
                         "hill": round(h, 5),
                         "u": round(float(losses_desc[k]), 5)})

    # GPD stability (xi, beta*) across thresholds
    stability = []
    for p in np.linspace(62, 88, 32):
        u   = np.percentile(losses, p)
        exc = losses[losses > u] - u
        if len(exc) >= 12:
            try:
                shape, _, scale = genpareto.fit(exc, floc=0)
                stability.append({
                    "u":        round(float(u), 5),
                    "xi":       round(float(shape), 5),
                    "beta":     round(float(scale), 5),
                    "beta_star": round(float(scale - shape * u), 5),
                    "n":        int(len(exc)),
                })
            except Exception:
                pass

    # Select u* from stability plateau
    u_star  = float(np.percentile(losses, 70))
    u_pct_v = 70.0
    if len(stability) >= 5:
        xi_arr = np.array([d["xi"] for d in stability])
        u_arr  = np.array([d["u"]  for d in stability])
        stds   = [xi_arr[i:i+5].std() for i in range(len(xi_arr) - 4)]
        idx_s  = int(np.argmin(stds)) + 2
        u_star  = float(u_arr[min(idx_s, len(u_arr) - 1)])
        u_pct_v = float(np.mean(losses <= u_star) * 100)

    # Fit GPD at u*
    exc_u  = losses[losses > u_star] - u_star
    zeta_u = len(exc_u) / n
    gpd = risk = None

    if len(exc_u) >= 12:
        try:
            xi, _, beta = genpareto.fit(exc_u, floc=0)
            gpd = {
                "xi":       round(float(xi), 4),
                "beta":     round(float(beta), 4),
                "u_star":   round(u_star, 5),
                "u_pct":    round(u_pct_v, 1),
                "n_exceed": int(len(exc_u)),
                "n_total":  int(n),
                "zeta_u":   round(float(zeta_u), 5),
            }
            # VaR & ES at 99%
            alpha = 0.99
            if abs(xi) > 1e-6:
                var99 = u_star + (beta / xi) * ((zeta_u / (1 - alpha)) ** xi - 1)
            else:
                var99 = u_star - beta * np.log(zeta_u / (1 - alpha))
            es99  = var99 / (1 - xi) + (beta - xi * u_star) / (1 - xi)
            risk  = {"var99": round(float(var99), 4),
                     "es99":  round(float(es99),  4)}
        except Exception:
            pass

    # Tail asymmetry: loss tail xi vs gain tail xi
    tail_asym = None
    gain_arr  = returns.dropna().values
    gain_arr  = gain_arr[gain_arr > 0]
    if len(gain_arr) >= 20:
        try:
            lq    = np.percentile(losses, 85)
            gq    = np.percentile(gain_arr, 85)
            l_exc = losses[losses > lq] - lq
            g_exc = gain_arr[gain_arr > gq] - gq
            if len(l_exc) >= 8 and len(g_exc) >= 8:
                xi_l, _, _ = genpareto.fit(l_exc, floc=0)
                xi_g, _, _ = genpareto.fit(g_exc, floc=0)
                asym       = float(xi_l - xi_g)
                tail_asym  = {
                    "xi_loss":       round(float(xi_l), 4),
                    "xi_gain":       round(float(xi_g), 4),
                    "asymmetry":     round(asym, 4),
                    "interpretation": (
                        "Strong unidirectional loss" if asym > 0.2 else
                        "Moderate asymmetry"         if asym > 0.05 else
                        "Near-symmetric (bidirectional)"
                    ),
                }
        except Exception:
            pass

    return {
        "mef":           mef,
        "hill":          hill,
        "stability":     stability,
        "gpd":           gpd,
        "risk":          risk,
        "tail_asymmetry": tail_asym,
    }


# ─── Performance metrics ──────────────────────────────────────────────────────

def perf(rets) -> dict:
    import numpy as np

    if len(rets) == 0:
        return {"cagr": 0, "total_return": 0, "sharpe": 0, "mdd": 0, "n_days": 0}
    cum_prod     = float((1 + rets).prod())
    cagr         = float(cum_prod ** (252 / max(len(rets), 1)) - 1)
    total_return = float(cum_prod - 1)
    sharpe       = float(rets.mean() / rets.std() * (252 ** 0.5)) if rets.std() > 0 else 0.0
    cum          = (1 + rets).cumprod()
    mdd          = float((cum / cum.cummax() - 1).min())
    return {
        "cagr":         round(cagr, 4),
        "total_return": round(total_return, 4),
        "sharpe":       round(sharpe, 3),
        "mdd":          round(mdd, 4),
        "n_days":       int(len(rets)),
    }


# ─── Straddle simulation ──────────────────────────────────────────────────────

def compute_straddle(prices, oos_start, target=0.10, horizon=120):
    import pandas as pd
    oos = prices[prices.index >= pd.Timestamp(oos_start)].iloc[:horizon]
    if len(oos) < 2:
        return None
    p0 = float(oos.iloc[0])

    long_pnl = short_pnl = None
    long_day = short_day = len(oos) - 1

    for i, p in enumerate(oos):
        p = float(p)
        if long_pnl is None and p / p0 - 1 >= target:
            long_pnl = target
            long_day = i
        if short_pnl is None and p0 / p - 1 >= target:
            short_pnl = target
            short_day = i
        if long_pnl is not None and short_pnl is not None:
            break

    final_p = float(oos.iloc[-1])
    if long_pnl is None:
        long_pnl = final_p / p0 - 1
    if short_pnl is None:
        short_pnl = p0 / final_p - 1

    return {
        "long_pnl":     round(long_pnl, 4),
        "short_pnl":    round(short_pnl, 4),
        "combined_pnl": round((long_pnl + short_pnl) / 2, 4),
        "long_hit":     long_pnl == target,
        "short_hit":    short_pnl == target,
        "long_day":     int(long_day),
        "short_day":    int(short_day),
        "long_date":    oos.index[min(long_day, len(oos)-1)].strftime("%Y-%m-%d"),
        "short_date":   oos.index[min(short_day, len(oos)-1)].strftime("%Y-%m-%d"),
        "target":       target,
        "n_days":       len(oos),
        "path": [{"date": d.strftime("%Y-%m-%d"), "ret": round(float(p) / p0 - 1, 4)}
                 for d, p in zip(oos.index, oos)],
    }


def mc_straddle_sim(sigma_daily, drift_daily, target=0.10, horizon=120, n_sim=10000, seed=42):
    import numpy as np
    rng = np.random.default_rng(seed)
    log_drift = drift_daily - 0.5 * sigma_daily ** 2
    z = rng.standard_normal((n_sim, horizon))
    price_rels = np.exp(np.cumsum(log_drift + sigma_daily * z, axis=1))

    long_ret  = price_rels - 1
    short_ret = 1.0 / price_rels - 1
    long_hit  = np.any(long_ret  >= target, axis=1)
    short_hit = np.any(short_ret >= target, axis=1)

    lhd = np.where(long_hit,  np.argmax(long_ret  >= target, axis=1), horizon - 1)
    shd = np.where(short_hit, np.argmax(short_ret >= target, axis=1), horizon - 1)
    lp  = np.where(long_hit,  target, price_rels[np.arange(n_sim), lhd] - 1)
    sp  = np.where(short_hit, target, 1.0 / price_rels[np.arange(n_sim), shd] - 1)

    return {
        "e_combined":  round(float(((lp + sp) / 2).mean()), 4),
        "e_long":      round(float(lp.mean()), 4),
        "e_short":     round(float(sp.mean()), 4),
        "p_long_hit":  round(float(long_hit.mean()), 3),
        "p_short_hit": round(float(short_hit.mean()), 3),
        "p_both_hit":  round(float((long_hit & short_hit).mean()), 3),
        "sigma_annual":round(float(sigma_daily * 252 ** 0.5), 4),
        "drift_annual":round(float(drift_daily * 252), 4),
        "n_sim":       n_sim,
    }


# ─── Main analysis ────────────────────────────────────────────────────────────

def run(params: dict) -> dict:
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    import pandas as pd
    import datetime

    ticker    = params.get("ticker",     "XLK")
    start     = params.get("start",      "2019-01-01")
    oos_start = params.get("oos_start",  "2019-07-01")
    end       = params.get("end") or (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
    direction = params.get("direction",  "SHORT")
    evt_name  = params.get("event_name", "Custom Event")
    evt_type  = params.get("event_type", "UNKNOWN")

    data     = fetch(ticker, start, end)
    gate_df  = compute_gate(data, ticker, direction)

    oos_dt   = pd.Timestamp(oos_start)
    oos_mask = gate_df.index >= oos_dt
    is_mask  = gate_df.index <  oos_dt

    oos_gate = gate_df[oos_mask]
    is_gate  = gate_df[is_mask]
    oos_open = float(oos_gate["gate_open"].mean() * 100) if len(oos_gate) > 0 else 0.0
    is_open  = float(is_gate["gate_open"].mean()  * 100) if len(is_gate)  > 0 else 0.0

    # Timeseries for chart (daily, limit 2000 rows)
    ts = gate_df.reset_index()
    ts.columns = ["date"] + list(ts.columns[1:])
    ts["date"] = ts["date"].dt.strftime("%Y-%m-%d")
    records = []
    for r in ts[["date","gate_score","beta_comp","vrp_comp","dir_comp","gate_open"]].to_dict("records"):
        for k in ("gate_score","beta_comp","vrp_comp","dir_comp"):
            v = r[k]
            r[k] = None if (v != v) else round(float(v), 4)
        r["gate_open"] = bool(r["gate_open"])
        records.append(r)

    # Returns
    sec_ret = data[ticker].pct_change().dropna()
    spy_ret = data["SPY"].pct_change().dropna()
    oos_sec = sec_ret[sec_ret.index >= oos_dt]
    oos_spy = spy_ret[spy_ret.index >= oos_dt]

    # Cumulative returns (sector vs SPY, OOS only)
    cum_sec = (1 + oos_sec).cumprod() - 1
    cum_spy = (1 + oos_spy).cumprod() - 1
    cum_ts  = []
    for dt in cum_sec.index:
        row = {"date": dt.strftime("%Y-%m-%d")}
        row["sector"] = round(float(cum_sec.loc[dt]), 4)
        if dt in cum_spy.index:
            row["spy"] = round(float(cum_spy.loc[dt]), 4)
        cum_ts.append(row)

    # Performance: all OOS vs gate-open only
    go_mask   = gate_df["gate_open"].reindex(oos_sec.index, fill_value=False)
    gated_ret = oos_sec[go_mask]

    # EVT
    evt = compute_evt(oos_sec)

    # Straddle simulation (entered on oos_start) + MC (pre-event vol from IS tail)
    straddle_sim = compute_straddle(data[ticker], oos_start)
    is_rets    = sec_ret[sec_ret.index < oos_dt]
    sigma_pre  = float(is_rets.tail(20).std()) if len(is_rets) >= 20 else float(oos_sec.std())
    drift_pre  = float(is_rets.tail(20).mean()) if len(is_rets) >= 20 else 0.0
    mc_result  = mc_straddle_sim(sigma_daily=sigma_pre, drift_daily=drift_pre)

    return {
        "event": {
            "name":      evt_name,
            "ticker":    ticker,
            "direction": direction,
            "type":      evt_type,
            "start":     start,
            "oos_start": oos_start,
            "end":       end,
        },
        "gate": {
            "params": {
                "thresh": GATE_THRESH,
                "consec": GATE_CONSEC,
                "window": GATE_WINDOW,
                "weights": {"beta": 0.40, "vrp": 0.35, "dir": 0.25},
            },
            "oos_open_pct": round(oos_open, 1),
            "is_open_pct":  round(is_open,  1),
            "timeseries":   records,
        },
        "performance": {
            "all_oos":   perf(oos_sec),
            "gate_open": perf(gated_ret),
            "bnh_spy":   perf(oos_spy),
        },
        "evt": evt,
        "returns": cum_ts[:600],
        "straddle": {
            "simulation": straddle_sim,
            "mc":         mc_result,
        },
    }


# ─── Vercel handler ───────────────────────────────────────────────────────────

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self._send(200, {})

    def do_POST(self):
        try:
            n    = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n)) if n else {}
            self._send(200, run(body))
        except Exception as e:
            self._send(500, {"error": str(e), "trace": traceback.format_exc()})

    def _send(self, code: int, data: dict):
        payload = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type",  "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_):
        pass

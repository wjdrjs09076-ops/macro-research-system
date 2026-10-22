"""
h1_sensitivity_test.py — H1 사전등록 검정: 회귀로 고른 섹터 금리 민감도의 표본 외 지속성.

규칙·구간·기준은 h1_sensitivity_preregistration.md 에 고정 — 상수를 바꾸면 새 사전등록.

    python h1_sensitivity_test.py fetch          # yfinance 재수집 → output/h1_levels_hist.csv (등급 C 원자료)
    python h1_sensitivity_test.py run            # 등급 C 판정 + 등급 A(오염, 보고만)·B(전향) 집계
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_loader as dl
from config import BENCHMARK, MACRO_TICKERS, OUTPUT_DIR, SECTOR_ETFS
from ontology.inference import T
from regression_engine import run_multivariate_regression

W_SEL       = 370                     # 등급 C 선정 롤링 창 (프리즈 시점 운영 n)
W_BETA      = 126                     # = kinetic HEDGE_BETA_WINDOW
MACROS      = ["VIX", "US10Y", "US2Y", "OIL", "DXY"]
GRADE_C     = ("2003-01-01", "2024-12-31")
HALF_SPLIT  = "2014-01-01"
GRADE_A     = ("2026-06-16", "2026-10-20")
GRADE_B_START, GRADE_B_N = "2026-10-21", 250
PASS_T      = 2.0
NW_LAG      = 5
FETCH_RANGE = ("1999-01-01", "2025-01-01")

HIST_CSV    = OUTPUT_DIR / "h1_levels_hist.csv"
LKG_CSV     = OUTPUT_DIR / "lkg_levels.csv"
RESULT_JSON = OUTPUT_DIR / "h1_sensitivity_result.json"


# ── 데이터 ────────────────────────────────────────────────────────────────────

def fetch_hist(attempts: int = 3) -> pd.DataFrame:
    import yfinance as yf
    tickers = list(SECTOR_ETFS) + [BENCHMARK] + list(MACRO_TICKERS.values())
    out = None
    for _ in range(attempts):                      # 시도끼리 칸 단위 합집합 (간헐 누락 대비)
        raw = yf.download(tickers, start=FETCH_RANGE[0], end=FETCH_RANGE[1],
                          auto_adjust=True, progress=False)["Close"]
        out = raw if out is None else out.combine_first(raw)
    out = out.rename(columns={v: k for k, v in MACRO_TICKERS.items()})
    out.index = pd.DatetimeIndex(out.index).tz_localize(None).normalize()
    return out[list(SECTOR_ETFS) + [BENCHMARK] + list(MACRO_TICKERS)]


def assemble(levels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """운영 정렬 규칙(data_loader._assemble) 그대로. 섹터는 상장 전이라도 열 유지."""
    px = levels[[c for c in levels.columns if c not in MACRO_TICKERS]].dropna(subset=[BENCHMARK])
    mc = levels[[c for c in levels.columns if c in MACRO_TICKERS]]
    ret, mchg, *_ = dl._assemble(px, mc)
    return ret, mchg


# ── 운영 선정 규칙 재현 ──────────────────────────────────────────────────────

def _bh(ps: list[float]) -> list[float]:
    finite = [i for i, p in enumerate(ps) if math.isfinite(p)]
    m = len(finite)
    qs, prev = [float("nan")] * len(ps), 1.0
    for rank, i in zip(range(m, 0, -1), sorted(finite, key=lambda i: ps[i])[::-1]):
        prev = min(prev, ps[i] * m / rank)
        qs[i] = prev
    return qs


def select(ret_w: pd.DataFrame, mchg_w: pd.DataFrame) -> dict[str, float]:
    """rate_victim → −1, rate_beneficiary → +1 (populate FDR + inference 임계와 동일)."""
    p = run_multivariate_regression(ret_w, mchg_w, standardize=True)
    p = p.reset_index() if "sector" not in p.columns else p
    fam = p[p.macro.isin(MACROS)].copy()
    t = pd.to_numeric(fam["t_delta_ctrl"], errors="coerce")
    fam["q"] = _bh([math.erfc(abs(x) / math.sqrt(2.0)) if math.isfinite(x) else float("nan") for x in t])
    g = fam.set_index(["sector", "macro"])
    w: dict[str, float] = {}
    for s in g.index.get_level_values(0).unique():
        def ok(m, cond):
            if (s, m) not in g.index:
                return False
            r = g.loc[(s, m)]
            d, tt, q = float(r.delta_ctrl), float(r.t_delta_ctrl), float(r.q)
            return all(map(math.isfinite, (d, tt, q))) and cond(d, tt) and q < 0.10
        v = 0.0
        if ok("US10Y", lambda d, tt: d < T["rate_victim_delta_max"] and tt < 0):
            v -= 1.0
        if any(ok(m, lambda d, tt: d > T["rate_benefit_delta_min"] and tt > 0) for m in ("US10Y", "US2Y")):
            v += 1.0
        if v:
            w[s] = v
    return w


def daily_panel(ret: pd.DataFrame, mchg: pd.DataFrame, start: str, end: str,
                expanding: bool) -> pd.DataFrame:
    """거래일 d 마다 d−1 까지로 선정·β → P_d, ΔUS10Y_d."""
    dates = ret.index
    i0, i1 = int(dates.searchsorted(pd.Timestamp(start))), int(dates.searchsorted(pd.Timestamp(end), "right"))
    rows = []
    for i in range(max(i0, W_BETA if expanding else W_SEL), i1):
        lo = 0 if expanding else i - W_SEL
        w = select(ret.iloc[lo:i], mchg.iloc[lo:i])
        if not w:
            continue
        hist = ret.iloc[i - W_BETA:i]
        spy = hist[BENCHMARK]
        legs = []
        for s, ws in w.items():
            b = float(hist[s].cov(spy) / spy.var())
            h = float(ret[s].iloc[i] - b * ret[BENCHMARK].iloc[i])
            if math.isfinite(h):
                legs.append((s, ws * h))
        dy = float(mchg["US10Y"].iloc[i])
        if legs and math.isfinite(dy):
            rows.append({"date": dates[i], "P": float(np.mean([v for _, v in legs])), "dUS10Y": dy,
                         "n_sel": len(legs), "legs": {s: v for s, v in legs}})
    return pd.DataFrame(rows)


def k_stats(df: pd.DataFrame) -> dict:
    if len(df) < 10:
        return {"n_days": int(len(df))}
    x, y = df["dUS10Y"].to_numpy(), df["P"].to_numpy()
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    n = len(y)
    s2 = e @ e / (n - 2)
    se = math.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    xc = x - x.mean()
    score = xc * e                                  # NW t: k 의 HAC 표준오차
    var = score @ score / n
    for L in range(1, NW_LAG + 1):
        var += 2.0 * (1 - L / (NW_LAG + 1)) * (score[L:] @ score[:-L]) / n
    se_nw = math.sqrt(var * n) / (xc @ xc) if var > 0 else float("nan")
    return {"n_days": n, "first": df.date.iloc[0].date().isoformat(), "last": df.date.iloc[-1].date().isoformat(),
            "k": round(float(beta[1]), 6), "t": round(float(beta[1] / se), 3),
            "t_nw_lag5": round(float(beta[1] / se_nw), 3), "avg_selected": round(float(df.n_sel.mean()), 2)}


def per_sector(df: pd.DataFrame) -> dict:
    out = {}
    legs = pd.DataFrame([{"date": r.date, "dUS10Y": r.dUS10Y, "s": s, "v": v}
                         for r in df.itertuples() for s, v in r.legs.items()])
    for s, g in legs.groupby("s"):
        out[s] = k_stats(g.rename(columns={"v": "P"}).assign(n_sel=1))
    return out


def run() -> dict:
    ret, mchg = assemble(pd.read_csv(HIST_CSV, index_col=0, parse_dates=True, float_precision="round_trip"))
    c = daily_panel(ret, mchg, *GRADE_C, expanding=False)
    res = {"preregistration": "h1_sensitivity_preregistration.md",
           "grade_C": k_stats(c),
           "grade_C_first_half": k_stats(c[c.date < pd.Timestamp(HALF_SPLIT)]),
           "grade_C_second_half": k_stats(c[c.date >= pd.Timestamp(HALF_SPLIT)]),
           "grade_C_per_sector_exploratory": per_sector(c)}
    g, h1, h2 = res["grade_C"], res["grade_C_first_half"], res["grade_C_second_half"]
    passed = (g.get("t") is not None and g["k"] > 0 and g["t"] >= PASS_T
              and h1.get("k", -1) > 0 and h2.get("k", -1) > 0)
    res["verdict_grade_C"] = "PASS" if passed else "REJECT"
    res["criterion"] = f"grade C: k>0 & t>={PASS_T} & 전반·후반 k>0"

    lret, lmchg = assemble(pd.read_csv(LKG_CSV, index_col=0, parse_dates=True, float_precision="round_trip"))
    res["grade_A_contaminated_not_evidence"] = k_stats(daily_panel(lret, lmchg, *GRADE_A, expanding=True))
    b = daily_panel(lret, lmchg, GRADE_B_START, "2100-01-01", expanding=True).head(GRADE_B_N)
    res["grade_B_forward"] = {**k_stats(b), "target_days": GRADE_B_N,
                              "status": "판정" if len(b) >= GRADE_B_N else f"진행 중 ({len(b)}/{GRADE_B_N})"}
    return res


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        lv = fetch_hist()
        lv.to_csv(HIST_CSV)
        print(f"saved {HIST_CSV} {lv.shape} {lv.index.min().date()}..{lv.index.max().date()}")
        print("NaN cells per column:", lv.isna().sum().to_dict())
    else:
        res = run()
        RESULT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(res, ensure_ascii=False, indent=2))

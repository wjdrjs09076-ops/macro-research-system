"""
h3_risk_test.py — H3 사전등록 검정: 장치의 섹터 금리 민감도가 단순 대안보다 나은 리스크 측정 도구인가.

규칙·구간·기준은 h3_risk_preregistration.md 에 고정 — 상수를 바꾸면 새 사전등록.

    python h3_risk_test.py [levels_csv]     # 기본 = output/h1_levels_hist.csv (H1 등급 C 원자료)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BENCHMARK, OUTPUT_DIR, SECTOR_ETFS
from h1_sensitivity_test import HIST_CSV, assemble, select   # H1 과 같은 원자료·운영 선정 규칙

W_SEL, W_BETA = 370, 126
MIN_BETA      = 100                   # β: 직전 126 중 유효 관측
MIN_SLOPE     = 250                   # slope: 직전 370 중 유효 쌍
SHOCK_Q       = 0.90                  # 충격일: |ΔUS10Y_d| ≥ 직전 370 |ΔUS10Y| 의 90% 분위
TEXTBOOK      = {"XLU": -1.0, "XLRE": -1.0, "XLF": +1.0}
GRADE_C       = ("2003-01-01", "2024-12-31")
HALF_SPLIT    = "2014-01-01"
PASS_T        = 2.0
NW_LAG        = 5
BLOCK, BLOCK_LAG, BLOCK_MIN = 63, 2, 40
MODELS        = ("zero", "textbook", "naive", "apparatus")
RESULT_JSON   = OUTPUT_DIR / "h3_risk_result.json"


# ── 일별 패널 ────────────────────────────────────────────────────────────────

def daily_panel(ret: pd.DataFrame, mchg: pd.DataFrame, start: str, end: str
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """거래일 d 마다 d−1 까지로 β·slope·선정 → 모델별 손실. (일별 표, 섹터×일 긴 표)"""
    mchg = mchg.reindex(ret.index)
    dates = ret.index
    sectors = [s for s in SECTOR_ETFS if s in ret.columns]
    R, M, X = ret[sectors].to_numpy(), ret[BENCHMARK].to_numpy(), mchg["US10Y"].to_numpy()
    i0, i1 = int(dates.searchsorted(pd.Timestamp(start))), int(dates.searchsorted(pd.Timestamp(end), "right"))
    rows, long = [], []
    for i in range(max(i0, W_SEL), i1):
        lo, dy = i - W_SEL, X[i]
        xw = X[lo:i]
        if not math.isfinite(dy) or np.isfinite(xw).sum() < MIN_SLOPE:
            continue
        thr = float(np.nanquantile(np.abs(xw), SHOCK_Q))
        w_app = select(ret.iloc[lo:i], mchg.iloc[lo:i])
        legs = []
        for j, s in enumerate(sectors):
            rb, mb = R[i - W_BETA:i, j], M[i - W_BETA:i]
            ok = np.isfinite(rb) & np.isfinite(mb)
            if ok.sum() < MIN_BETA:
                continue
            beta = float(np.cov(rb[ok], mb[ok])[0, 1] / np.var(mb[ok], ddof=1))
            y = float(R[i, j] - beta * M[i])
            hw = R[lo:i, j] - beta * M[lo:i]
            ok2 = np.isfinite(hw) & np.isfinite(xw)
            if not math.isfinite(y) or ok2.sum() < MIN_SLOPE:
                continue
            slope = float(np.cov(hw[ok2], xw[ok2])[0, 1] / np.var(xw[ok2], ddof=1))
            b = {"zero": 0.0, "naive": slope,
                 "textbook": TEXTBOOK.get(s, 0.0) * abs(slope),
                 "apparatus": w_app.get(s, 0.0) * abs(slope)}
            legs.append((s, y, {m: b[m] * dy for m in MODELS}))
        if not legs:
            continue
        shock = abs(dy) >= thr
        n_sel = sum(1 for s, _, _ in legs if w_app.get(s, 0.0) != 0.0)
        row = {"date": dates[i], "dUS10Y": float(dy), "shock": bool(shock),
               "n_avail": len(legs), "n_sel": n_sel, "G": n_sel / len(legs)}
        for m in MODELS:
            row[f"L_{m}"] = float(np.mean([(y - p[m]) ** 2 for _, y, p in legs]))
            row[f"A_{m}"] = float(np.mean([abs(y - p[m]) for _, y, p in legs]))
        rows.append(row)
        long += [{"date": dates[i], "s": s, "y": y, "dUS10Y": float(dy), "shock": bool(shock),
                  **{f"e_{m}": (y - p[m]) ** 2 for m in MODELS}} for s, y, p in legs]
    return pd.DataFrame(rows), pd.DataFrame(long)


# ── 통계 ─────────────────────────────────────────────────────────────────────

def ols_nw(y: np.ndarray, X: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS 계수, 단순 t, Newey-West t."""
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ y
    e = y - X @ beta
    t = beta / np.sqrt(np.diag(XtXi) * (e @ e) / (n - k))
    g = X * e[:, None]
    S = g.T @ g
    for L in range(1, lag + 1):
        G = g[L:].T @ g[:-L]
        S += (1 - L / (lag + 1)) * (G + G.T)
    V = XtXi @ S @ XtXi
    return beta, t, beta / np.sqrt(np.diag(V))


def dm(d: pd.Series) -> dict:
    d = d.to_numpy(dtype=float)
    if len(d) < 10:
        return {"n": int(len(d))}
    if np.ptp(d) == 0:                                  # 두 모델 예측이 매일 같음 → 차이 없음(검정 불가)
        return {"n": int(len(d)), "mean": float(d.mean()), "t": None, "t_nw": None}
    _, t, t_nw = ols_nw(d, np.ones((len(d), 1)), NW_LAG)
    return {"n": int(len(d)), "mean": float(d.mean()), "t": round(float(t[0]), 3), "t_nw": round(float(t_nw[0]), 3)}


def es(df: pd.DataFrame, prefix: str = "L_") -> dict:
    z = df[f"{prefix}zero"].sum()
    return {m: round(float(1 - df[f"{prefix}{m}"].sum() / z), 4) for m in MODELS if m != "zero"}


def comparisons(df: pd.DataFrame, prefix: str = "L_") -> dict:
    out = {}
    for base in ("textbook", "naive"):
        d = df[f"{prefix}{base}"] - df[f"{prefix}apparatus"]
        first, second = df.date < pd.Timestamp(HALF_SPLIT), df.date >= pd.Timestamp(HALF_SPLIT)
        out[f"{base}_minus_apparatus"] = {"all": dm(d), "first_half": dm(d[first]), "second_half": dm(d[second])}
    return out


def passes(c: dict) -> bool:
    ok = True
    for v in c.values():
        a, h1, h2 = v["all"], v["first_half"], v["second_half"]
        ok &= (a.get("mean", -1) > 0 and (a.get("t_nw") or -1) >= PASS_T
               and h1.get("mean", -1) > 0 and h2.get("mean", -1) > 0)
    return bool(ok)


# ── 레짐 (탐색) ──────────────────────────────────────────────────────────────

def regime(p: pd.DataFrame, long: pd.DataFrame) -> dict:
    yearly = {}
    for yr, g in p.groupby(p.date.dt.year):
        sh = g[g.shock]
        yearly[int(yr)] = {"mean_G": round(float(g.G.mean()), 3), "n_shock": int(len(sh)),
                           **({f"ES_{k}": v for k, v in es(sh).items()} if len(sh) else {})}
    blocks = []
    dates = list(p.date)
    for b0 in range(0, len(dates) - BLOCK + 1, BLOCK):
        span = dates[b0:b0 + BLOCK]
        seg = long[long.date.isin(span)]
        r2 = []
        for _, g in seg.groupby("s"):
            g = g[np.isfinite(g.y)]
            if len(g) >= BLOCK_MIN and g.dUS10Y.var() > 0:
                r2.append(float(np.corrcoef(g.y, g.dUS10Y)[0, 1] ** 2))
        if r2:
            blocks.append({"start": span[0], "G": float(p.G.iloc[b0]), "RI": float(np.mean(r2))})
    bl = pd.DataFrame(blocks)
    bl["RI_prev"] = bl.RI.shift(1)
    out = {"yearly": yearly, "n_blocks": int(len(bl))}
    if len(bl) >= 10:
        b, t, tn = ols_nw(bl.RI.to_numpy(), np.column_stack([np.ones(len(bl)), bl.G]), BLOCK_LAG)
        out["RI_on_G"] = {"corr": round(float(bl.RI.corr(bl.G)), 3), "coef_G": round(float(b[1]), 4),
                          "t": round(float(t[1]), 3), "t_nw": round(float(tn[1]), 3)}
        q = bl.dropna()
        b, t, tn = ols_nw(q.RI.to_numpy(), np.column_stack([np.ones(len(q)), q.G, q.RI_prev]), BLOCK_LAG)
        out["RI_on_G_and_RI_prev"] = {"n": int(len(q)), "coef_G": round(float(b[1]), 4), "t_G": round(float(t[1]), 3),
                                      "t_nw_G": round(float(tn[1]), 3), "coef_RI_prev": round(float(b[2]), 4),
                                      "t_nw_RI_prev": round(float(tn[2]), 3)}
    return out


# ── 실행 ─────────────────────────────────────────────────────────────────────

def run(levels_csv: Path = HIST_CSV) -> dict:
    ret, mchg = assemble(pd.read_csv(levels_csv, index_col=0, parse_dates=True, float_precision="round_trip"))
    p, long = daily_panel(ret, mchg, *GRADE_C)
    sh = p[p.shock]
    c = comparisons(sh)
    per_sector = {}
    for s, g in long[long.shock].groupby("s"):
        z = g.e_zero.sum()
        per_sector[s] = {"n": int(len(g)), **{f"ES_{m}": round(float(1 - g[f"e_{m}"].sum() / z), 4)
                                              for m in MODELS if m != "zero"}}
    return {
        "preregistration": "h3_risk_preregistration.md",
        "window": {"first": p.date.iloc[0].date().isoformat(), "last": p.date.iloc[-1].date().isoformat(),
                   "n_days": int(len(p)), "n_shock": int(len(sh)), "avg_avail": round(float(p.n_avail.mean()), 2),
                   "avg_selected": round(float(p.n_sel.mean()), 2)},
        "shock_comparisons": c,
        "verdict": "PASS" if passes(c) else "REJECT",
        "criterion": f"두 비교 모두: 평균 D>0 & NW(lag {NW_LAG}) t>={PASS_T} & 전·후반 평균 D>0 (충격일, 제곱오차)",
        "report_shock_ES": es(sh),
        "report_shock_abs_comparisons": comparisons(sh, "A_"),
        "report_all_days_ES": es(p),
        "report_all_days_comparisons": comparisons(p),
        "report_shock_per_sector_ES": per_sector,
        "report_regime_exploratory": regime(p, long),
    }


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    res = run(Path(sys.argv[1]) if len(sys.argv) > 1 else HIST_CSV)
    RESULT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2, default=str))

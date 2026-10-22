"""
h2_slope_test.py — H2-2 사전등록 검정: 기간 스프레드(10Y−1Y) 실시간 예측회귀 → 10년물 방향.

규칙·구간·판정 기준·정지 규칙은 h2_slope_preregistration.md 에 고정돼 있다 — 상수를 바꾸면
새 사전등록이고, 정지 규칙상 H2 에서는 더 이상 허용되지 않는다.

    python h2_slope_test.py <h15_tcm_package.csv>
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUT_DIR
from h2_tsmom_test import _nw_t

INPUTS_CSV  = OUTPUT_DIR / "h2_slope_inputs.csv"
RESULT_JSON = OUTPUT_DIR / "h2_slope_result.json"
LONG_ID, SHORT_ID = "H15/H15/RIFLGFCY10_N.B", "H15/H15/RIFLGFCY01_N.B"

HOLD       = 21
MIN_TRAIN  = 2520                      # 학습쌍 ~10년
PRIMARY    = ("1992-01-01", "2026-10-20")
HALF_SPLIT = "2009-01-01"
PREPUB     = ("1972-01-01", "1991-12-31")
PASS_T     = 2.24                      # Bonferroni: H2 규칙 2개, 양측 5%
NW_LAG     = 3


def load_h15(path: Path) -> pd.DataFrame:
    rows = list(csv.reader(path.open(encoding="utf-8-sig", newline="")))
    ids = next(r for r in rows if r and r[0] == "Unique Identifier:")
    c10, c1 = ids.index(LONG_ID), ids.index(SHORT_ID)
    start = next(i for i, r in enumerate(rows) if r and r[0] == "Time Period") + 1
    df = pd.DataFrame([(r[0], r[c10], r[c1]) for r in rows[start:]], columns=["date", "y10", "y1"])
    df["date"] = pd.to_datetime(df["date"])
    for c in ("y10", "y1"):
        df[c] = pd.to_numeric(df[c], errors="coerce")       # ND → NaN
    return df.dropna().set_index("date").sort_index()


def blocks(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    dates = df.index
    y10, x = df["y10"].to_numpy(), (df["y10"] - df["y1"]).to_numpy()
    n = len(y10)
    dy = np.full(n, np.nan)
    dy[:n - HOLD] = (y10[HOLD:] - y10[:n - HOLD]) * 100.0     # Δ_j, 결과 시점 j+HOLD
    i = int(dates.searchsorted(pd.Timestamp(start)))
    end_ts = pd.Timestamp(end)
    rows = []
    while i + HOLD < n and dates[i + HOLD] <= end_ts:
        m = i - HOLD + 1                                     # 학습쌍 j ∈ [0, m): j + HOLD ≤ i
        s, b, f = 0.0, np.nan, np.nan
        if m >= MIN_TRAIN:
            xt, zt = x[:m], dy[:m]
            b, a = np.polyfit(xt, zt, 1)
            f = a + b * x[i]
            s = float(np.sign(f))
        rows.append({"start": dates[i], "end": dates[i + HOLD], "spread": x[i], "beta": b,
                     "forecast_bp": f, "signal": s, "dy_bp": dy[i], "r": s * dy[i]})
        i += HOLD
    return pd.DataFrame(rows)


def stats(b: pd.DataFrame) -> dict:
    b = b[b.signal != 0]
    r = b["r"].to_numpy(dtype=float)
    n = len(r)
    if n < 2:
        return {"n": n}
    sd = r.std(ddof=1)
    live = b[b.dy_bp != 0]
    return {
        "n": n,
        "first_block": b["start"].iloc[0].date().isoformat(),
        "last_block_end": b["end"].iloc[-1].date().isoformat(),
        "mean_bp": round(float(r.mean()), 3),
        "sd_bp": round(float(sd), 3),
        "t": round(float(r.mean() / sd * np.sqrt(n)), 3) if sd > 0 else None,
        "t_nw_lag3": round(_nw_t(r, NW_LAG), 3),
        "hit_rate": round(float((live["r"] > 0).mean()), 4) if len(live) else None,
        "share_beta_negative": round(float((b.beta < 0).mean()), 4),
        "share_forecast_up": round(float((b.signal > 0).mean()), 4),
    }


def run(df: pd.DataFrame) -> dict:
    prim = blocks(df, *PRIMARY)
    res = {
        "preregistration": "h2_slope_preregistration.md",
        "rule": f"x=y10-y1, 확장창 OLS Δ(next {HOLD}obs, bp)=a+b·x (j+{HOLD}<=i 만), s=sign(a+b·x_i), r=s·Δ_i",
        "data": {"series": f"{LONG_ID} − {SHORT_ID}", "first_obs": df.index[0].date().isoformat(),
                 "last_obs": df.index[-1].date().isoformat(), "n_obs": int(len(df))},
        "primary": stats(prim),
        "first_half": stats(prim[prim.start < pd.Timestamp(HALF_SPLIT)]),
        "second_half": stats(prim[prim.start >= pd.Timestamp(HALF_SPLIT)]),
        "prepub_reference_not_evidence": stats(blocks(df, *PREPUB)),
    }
    p, h1, h2 = res["primary"], res["first_half"], res["second_half"]
    passed = (p.get("t") is not None and p["t"] >= PASS_T
              and h1.get("mean_bp", -1) > 0 and h2.get("mean_bp", -1) > 0)
    res["verdict"] = "PASS" if passed else "REJECT"
    res["criterion"] = f"primary t >= {PASS_T} (Bonferroni, H2 규칙 2개) AND 전반·후반 mean_bp > 0"
    res["stopping_rule"] = "H2-2 = H2 마지막 시도. REJECT 면 H2 종결."
    if p.get("t") is not None and (p["t"] >= PASS_T) != (p["t_nw_lag3"] >= PASS_T):
        res["note"] = "simple t 와 NW t 판정 엇갈림 — 판정은 사전등록 기준(simple t)대로"
    return res


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    src = Path(sys.argv[1])
    df = load_h15(src)
    df.to_csv(INPUTS_CSV)
    res = run(df)
    res["data"]["raw_sha256"] = hashlib.sha256(src.read_bytes()).hexdigest()
    RESULT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))

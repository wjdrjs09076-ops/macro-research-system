"""
h2_tsmom_test.py — H2-1 사전등록 검정: 미국 10년물 금리 시계열 모멘텀.

규칙·구간·판정 기준은 h2_tsmom_preregistration.md 에 고정돼 있다 — 이 파일의 상수를 바꾸면
그건 새 사전등록이다(시도 규칙 수 +1, 다중검정으로 셈).

    python h2_tsmom_test.py               # FRED 공개 CSV 다운로드 → output/h2_dgs10.csv 저장 후 검정
    python h2_tsmom_test.py path/to.csv   # 저장된 CSV 로 재현
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUT_DIR

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
DATA_CSV     = OUTPUT_DIR / "h2_dgs10.csv"
RESULT_JSON  = OUTPUT_DIR / "h2_tsmom_result.json"

LOOKBACK   = 252                       # 12개월 (MOP 2012)
HOLD       = 21                        # 1개월
PRIMARY    = ("2013-01-01", "2026-10-20")
HALF_SPLIT = "2020-01-01"
PREPUB     = ("1962-01-01", "2012-12-31")
PASS_T     = 2.0
NW_LAG     = 3


def load_dgs10(path: str | None = None) -> pd.Series:
    if path is None:
        raw = urllib.request.urlopen(FRED_CSV_URL, timeout=60).read()
        DATA_CSV.write_bytes(raw)
        path = DATA_CSV
    df = pd.read_csv(path)
    date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
    y = pd.to_numeric(df["DGS10"], errors="coerce")
    y.index = pd.to_datetime(df[date_col])
    return y.dropna().sort_index()


def blocks(y: pd.Series, start: str, end: str) -> pd.DataFrame:
    """겹치지 않는 HOLD 관측 블록. 블록 시작일이 [start, end], 블록 끝 관측도 end 이내."""
    dates, vals = y.index, y.to_numpy()
    i = max(int(dates.searchsorted(pd.Timestamp(start))), LOOKBACK)
    end_ts = pd.Timestamp(end)
    rows = []
    while i + HOLD < len(vals) and dates[i + HOLD] <= end_ts:
        s = float(np.sign(vals[i] - vals[i - LOOKBACK]))
        d = (vals[i + HOLD] - vals[i]) * 100.0
        rows.append({"start": dates[i], "end": dates[i + HOLD], "signal": s, "dy_bp": d, "r": s * d})
        i += HOLD
    return pd.DataFrame(rows)


def _nw_t(r: np.ndarray, lag: int) -> float:
    n = len(r)
    e = r - r.mean()
    var = e @ e / n
    for k in range(1, lag + 1):
        var += 2.0 * (1.0 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return float(r.mean() / np.sqrt(var / n)) if var > 0 else float("nan")


def stats(b: pd.DataFrame) -> dict:
    r = b["r"].to_numpy(dtype=float)
    n = len(r)
    if n < 2:
        return {"n": n}
    sd = r.std(ddof=1)
    live = b[(b.signal != 0) & (b.dy_bp != 0)]
    return {
        "n": n,
        "first_block": b["start"].iloc[0].date().isoformat(),
        "last_block_end": b["end"].iloc[-1].date().isoformat(),
        "mean_bp": round(float(r.mean()), 3),
        "sd_bp": round(float(sd), 3),
        "t": round(float(r.mean() / sd * np.sqrt(n)), 3) if sd > 0 else None,
        "t_nw_lag3": round(_nw_t(r, NW_LAG), 3),
        "hit_rate": round(float((live["r"] > 0).mean()), 4) if len(live) else None,
        "share_long_yield": round(float((b.signal > 0).mean()), 4),
    }


def run(y: pd.Series) -> dict:
    prim = blocks(y, *PRIMARY)
    first = prim[prim.start < pd.Timestamp(HALF_SPLIT)]
    second = prim[prim.start >= pd.Timestamp(HALF_SPLIT)]
    res = {
        "preregistration": "h2_tsmom_preregistration.md",
        "rule": f"s=sign(y_i - y_(i-{LOOKBACK})), r=s*(y_(i+{HOLD})-y_i)*100bp, 겹치지 않는 {HOLD}관측 블록",
        "data": {"series": "FRED DGS10", "first_obs": y.index[0].date().isoformat(),
                 "last_obs": y.index[-1].date().isoformat(), "n_obs": int(len(y))},
        "primary": stats(prim),
        "first_half": stats(first),
        "second_half": stats(second),
        "prepub_reference_not_evidence": stats(blocks(y, *PREPUB)),
    }
    p, h1, h2 = res["primary"], res["first_half"], res["second_half"]
    passed = (p.get("t") is not None and p["t"] >= PASS_T
              and h1.get("mean_bp", -1) > 0 and h2.get("mean_bp", -1) > 0)
    res["verdict"] = "PASS" if passed else "REJECT"
    res["criterion"] = f"primary t >= {PASS_T} AND 전반·후반 mean_bp > 0"
    if p.get("t") is not None and (p["t"] >= PASS_T) != (p["t_nw_lag3"] >= PASS_T):
        res["note"] = "simple t 와 NW t 판정 엇갈림 — 판정은 사전등록 기준(simple t)대로"
    return res


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    res = run(load_dgs10(sys.argv[1] if len(sys.argv) > 1 else None))
    RESULT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))

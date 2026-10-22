"""
h2_h15_to_csv.py — 연준 H.15(Data Download Program) 패키지 CSV → h2_tsmom_test.py 입력 형식.

H2-1 사전등록 정정 1 (h2_tsmom_preregistration.md): FRED 점검으로 원천을 FRED DGS10 의 원
발행처인 연준 H.15 `H15/H15/RIFLGFCY10_N.B`(같은 시리즈)로 바꾼다. 규칙·구간·기준 불변.

    python h2_h15_to_csv.py <h15_package.csv>   # → output/h2_dgs10.csv (observation_date,DGS10)
"""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUT_DIR

H15_URL   = ("https://www.federalreserve.gov/datadownload/Output.aspx?rel=H15&series="
             "bf17364827e38702b42a58cf8eaa3f78&lastobs=&from=&to=&filetype=csv&label=include"
             "&layout=seriescolumn&type=package")
SERIES_ID = "H15/H15/RIFLGFCY10_N.B"
OUT_CSV   = OUTPUT_DIR / "h2_dgs10.csv"


def convert(src: Path, dst: Path = OUT_CSV) -> dict:
    rows = list(csv.reader(src.open(encoding="utf-8-sig", newline="")))
    id_row = next(i for i, r in enumerate(rows) if r and r[0] == "Unique Identifier:")
    col = rows[id_row].index(SERIES_ID)
    data_start = next(i for i, r in enumerate(rows) if r and r[0] == "Time Period") + 1
    n_val = n_nd = 0
    with dst.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["observation_date", "DGS10"])
        for r in rows[data_start:]:
            v = r[col].strip()
            if v in ("ND", "NC", "NA", ""):
                n_nd += 1
                v = ""                        # h2_tsmom_test 가 결측으로 제거
            else:
                n_val += 1
            w.writerow([r[0], v])
    return {"source": H15_URL, "series": SERIES_ID,
            "raw_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "n_values": n_val, "n_missing": n_nd, "out": str(dst)}


if __name__ == "__main__":
    print(convert(Path(sys.argv[1])))

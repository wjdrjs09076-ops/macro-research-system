from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path

ROOT        = Path(__file__).resolve().parent
OUTPUT_DIR  = ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
ROLLING_DIR = OUTPUT_DIR / "rolling"

SHARADAR_API_KEY = os.environ.get("NASDAQ_DATA_LINK_KEY", "")

START_DATE = "2025-01-01"
# END_DATE 는 라이브 운영용 — yfinance 가 오늘까지 받도록 매번 동적 계산.
# (백테스트 일관성이 필요한 모듈은 자체 날짜 상수를 쓰므로 영향 없음.)
END_DATE   = _dt.date.today().isoformat()
OOS_START  = "2026-01-01"   # out-of-sample evaluation start

# 11 GICS sector SPDR ETFs
SECTOR_ETFS: dict[str, str] = {
    "XLK":  "Technology",
    "XLF":  "Financials",
    "XLE":  "Energy",
    "XLV":  "Health Care",
    "XLY":  "Consumer Discretionary",
    "XLP":  "Consumer Staples",
    "XLU":  "Utilities",
    "XLI":  "Industrials",
    "XLB":  "Materials",
    "XLRE": "Real Estate",
    "XLC":  "Communication Services",
}

BENCHMARK = "SPY"

# yfinance tickers for macro variables
# OIL (2026-06-12 추가): 금리 delta의 유가 교란 통제용. 호르무즈 표본에서
# 유가↑→금리↑ 동시 발생이 XLE delta(US10Y)>0 을 만들었을 가능성 — OIL 을
# 회귀에 동시 투입해야 partial delta 가 깨끗해진다.
MACRO_TICKERS: dict[str, str] = {
    "VIX":   "^VIX",
    "US10Y": "^TNX",
    "US2Y":  "^IRX",
    "DXY":   "DX-Y.NYB",
    "OIL":   "CL=F",
}

ROLLING_WINDOW = 63   # ~3 months
RV_WINDOW      = 20   # realized vol lookback (trading days)

"""
data_loader.py — Sharadar(SFP) 섹터 ETF 가격 + yfinance 매크로 변수 수집

섹터 ETF 수익률과 매크로 변수 변화량을 반환한다.

매크로 변화량 정의:
  VIX, US10Y, US2Y : 레벨 변화 (Δ in index/% points)
  DXY              : 퍼센트 수익률 (log return)
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import nasdaqdatalink
import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    SHARADAR_API_KEY,
    START_DATE,
    END_DATE,
    SECTOR_ETFS,
    BENCHMARK,
    MACRO_TICKERS,
    OUTPUT_DIR,
)

nasdaqdatalink.ApiConfig.api_key = SHARADAR_API_KEY


# ─────────────────────────────────────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_sfp_ticker(ticker: str) -> pd.Series:
    """SHARADAR/SFP에서 단일 ticker의 조정종가를 반환. 실패 시 yfinance fallback."""
    try:
        df = nasdaqdatalink.get_table(
            "SHARADAR/SFP",
            ticker=ticker,
            date={"gte": START_DATE, "lte": END_DATE},
            paginate=True,
        )
        if df.empty:
            raise ValueError("empty response")
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")["closeadj"].rename(ticker).sort_index()
    except Exception as e:
        print(f"  [SFP WARN] {ticker}: {e} → yfinance fallback")
        raw = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
        return raw["Close"].rename(ticker)


def _fetch_macro_levels() -> pd.DataFrame:
    """yfinance로 VIX, 10Y, 2Y, DXY 레벨 데이터 수집."""
    tickers = list(MACRO_TICKERS.values())
    raw = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

    # yfinance >= 0.2 returns MultiIndex columns
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw["Close"]
    else:
        raw = raw[["Close"]] if "Close" in raw.columns else raw

    rename = {v: k for k, v in MACRO_TICKERS.items()}
    return raw.rename(columns=rename).sort_index()


# ─────────────────────────────────────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_sector_prices() -> pd.DataFrame:
    """섹터 ETF + SPY 조정종가 DataFrame 반환 (date index, ticker columns)."""
    tickers = list(SECTOR_ETFS.keys()) + [BENCHMARK]
    series_list = []
    for tkr in tickers:
        print(f"  Fetching {tkr}...")
        series_list.append(_fetch_sfp_ticker(tkr))
    prices = pd.concat(series_list, axis=1).sort_index()
    return prices[(prices.index >= START_DATE) & (prices.index <= END_DATE)]


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """일간 로그 수익률 (소수점, annualize 전)."""
    return np.log(prices / prices.shift(1))


def compute_macro_changes(macro: pd.DataFrame) -> pd.DataFrame:
    """
    매크로 변화량:
      VIX, US10Y, US2Y → 레벨 1차 차분 (Δ)
      DXY              → 로그 수익률 (%)
    """
    chg = macro.diff().copy()
    # DXY는 퍼센트 변화로 덮어씀
    if "DXY" in macro.columns:
        chg["DXY"] = np.log(macro["DXY"] / macro["DXY"].shift(1))
    return chg


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    모든 데이터 로드 및 날짜 정렬.

    반환: (sector_returns, macro_changes, sector_prices, macro_levels)
      - sector_returns  : 일간 로그 수익률
      - macro_changes   : 매크로 변화량 (정의는 compute_macro_changes 참고)
      - sector_prices   : 원시 조정종가
      - macro_levels    : 원시 매크로 레벨
    """
    print("Fetching sector ETF prices from SHARADAR/SFP...")
    prices = fetch_sector_prices()

    print("Fetching macro variables from yfinance...")
    macro_lvl = _fetch_macro_levels()

    # 공통 날짜로 정렬
    common = prices.index.intersection(macro_lvl.index)
    prices    = prices.loc[common]
    macro_lvl = macro_lvl.loc[common]

    returns  = compute_log_returns(prices)
    macro_chg = compute_macro_changes(macro_lvl)

    # 1차 차분 후 공통 날짜 재정렬 (첫 행 NaN 제거)
    common2   = returns.index.intersection(macro_chg.index)
    returns   = returns.loc[common2].dropna(how="all")
    macro_chg = macro_chg.loc[common2].dropna(how="all")

    return returns, macro_chg, prices, macro_lvl


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ret, mchg, px, mlvl = load_all()
    ret.to_parquet(OUTPUT_DIR / "sector_returns.parquet")
    mchg.to_parquet(OUTPUT_DIR / "macro_changes.parquet")
    px.to_parquet(OUTPUT_DIR / "sector_prices.parquet")
    mlvl.to_parquet(OUTPUT_DIR / "macro_levels.parquet")
    print(f"\nSaved. Returns: {ret.shape}  Macro changes: {mchg.shape}")
    print(ret.tail(3))
    print(mchg.tail(3))

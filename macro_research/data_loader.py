"""
data_loader.py — Sharadar(SFP) 섹터 ETF 가격 + yfinance 매크로 변수 수집

섹터 ETF 수익률과 매크로 변수 변화량을 반환한다.

매크로 변화량 정의:
  VIX, US10Y, US2Y : 레벨 변화 (Δ in index/% points)
  DXY              : 퍼센트 수익률 (log return)
"""
from __future__ import annotations

import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

# nasdaqdatalink 는 PyPI 에서 제거됨 (현 패키지명 nasdaq-data-link) + Sharadar
# 구독 취소(2026-06) — 선택적 import. 미설치 시 yfinance 폴백 전용 (2026-06-12).
try:
    import nasdaqdatalink
except ImportError:
    nasdaqdatalink = None

from config import (
    SHARADAR_API_KEY,
    START_DATE,
    END_DATE,
    SECTOR_ETFS,
    BENCHMARK,
    MACRO_TICKERS,
    OUTPUT_DIR,
)

if nasdaqdatalink is not None:
    nasdaqdatalink.ApiConfig.api_key = SHARADAR_API_KEY


# ─────────────────────────────────────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_sfp_ticker(ticker: str) -> pd.Series:
    """SHARADAR/SFP에서 단일 ticker의 조정종가를 반환. 실패 시 yfinance fallback."""
    try:
        if nasdaqdatalink is None:
            raise ImportError("nasdaqdatalink 미설치 — yfinance 폴백")
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
        close = raw["Close"]
        # yfinance >= 0.2 는 단일 티커도 MultiIndex 컬럼 → Close 가 DataFrame
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close.rename(ticker)


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
      DXY, OIL         → 로그 수익률 (가격 시리즈)
    """
    chg = macro.diff().copy()
    # 가격 시리즈는 퍼센트 변화로 덮어씀
    for px_col in ("DXY", "OIL"):
        if px_col in macro.columns:
            chg[px_col] = np.log(macro[px_col] / macro[px_col].shift(1))
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

    return _assemble(prices, macro_lvl)


def _assemble(prices: pd.DataFrame, macro_lvl: pd.DataFrame
              ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """원시 레벨 → (returns, macro_changes, prices, macro_levels). load_all 과 같은 정렬 규칙."""
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


# ─────────────────────────────────────────────────────────────────────────────
# 라이브 경로: last-known-good(LKG) 원시 레벨 + 거래일 단위 고정 (2026-10-21)
# ─────────────────────────────────────────────────────────────────────────────
# yfinance 는 간헐적으로 불완전한 이력을 돌려준다. 9/02 CI 실측(런 33639312973 vs
# 33642529938): 8/28 행이 US10Y·US2Y·VIX·DXY·XLK·XLC 에서 빠진 응답과 완전한 응답이
# 사이클마다 번갈아 옴 → diff 로 2행 NaN → 회귀 표본 416↔414 → XLU rate_victim
# q 0.092↔0.127 → 하루 6회 진입↔은퇴. 아무 검증 없이 조용히 줄어든 표본으로 회귀했다.
#
# 원시 레벨은 '마지막 완료 거래일까지의 일봉'(END_DATE 배타)이라 장중에 바뀔 이유가
# 없다. 그래서 장중 변화는 정의상 데이터 결함이다:
#   advanced : fresh 에 새 거래일이 붙었다 → fresh 우선, fresh 의 빈 칸은 LKG 로 메움
#   pinned   : 같은 거래일 → LKG 값 고정. fresh 는 LKG 의 빈 칸만 채움(NaN→값 단조,
#              값→NaN·값→다른값 불가 = 왕복 불가)
#   lkg_only : fetch 전부 실패 → LKG 그대로
# LKG 는 Stage 5 에서 커밋된다(러너 휘발성 — 커밋 없이는 매 사이클 bootstrap).
# 한계: 배당락으로 수정주가 기준이 바뀐 날 LKG 로 메운 칸은 옛 기준 값(그 칸 하루
# 수익률이 배당률만큼 왜곡). 결측과 배당락이 같은 날 겹칠 때만 발생.
LKG_LEVELS_CSV    = OUTPUT_DIR / "lkg_levels.csv"
DATA_QUALITY_JSON = OUTPUT_DIR / "data_quality.json"
FETCH_ATTEMPTS    = 3
MIN_LEVEL_ROWS    = 60     # bootstrap 이 이보다 짧으면 fetch 붕괴로 보고 중단


def _fetch_levels() -> pd.DataFrame:
    """섹터 가격 + 매크로 레벨을 공통 날짜로 붙인 원시 레벨 (열 = 티커 + 매크로)."""
    prices = fetch_sector_prices()
    macro_lvl = _fetch_macro_levels()
    lv = prices.join(macro_lvl, how="inner").sort_index()
    idx = pd.DatetimeIndex(lv.index)
    lv.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    return lv


def _read_lkg() -> pd.DataFrame | None:
    if not LKG_LEVELS_CSV.exists():
        return None
    try:
        lkg = pd.read_csv(LKG_LEVELS_CSV, index_col=0, parse_dates=True,
                          float_precision="round_trip")
    except Exception as exc:
        print(f"  [LKG WARN] 읽기 실패 → LKG 없이 진행: {exc}")
        return None
    return lkg if len(lkg) else None


def _suspect_gaps(levels: pd.DataFrame, after: pd.Timestamp | None) -> list[str]:
    """`after` 이후 새 행에서, 바로 앞 행엔 값이 있는데 비어 있는 칸 = 응답 누락 의심."""
    prev = levels.shift(1).notna()
    hole = levels.isna() & prev
    if after is not None:
        hole = hole[hole.index > after]
    return [f"{d.date().isoformat()}:{c}"
            for d, row in hole.iterrows() for c in row.index[row.values]]


def _stable_levels() -> tuple[pd.DataFrame, dict]:
    lkg = _read_lkg()
    lkg_last = lkg.index.max() if lkg is not None else None

    fresh, err, attempts = None, None, 0
    for attempts in range(1, FETCH_ATTEMPTS + 1):
        try:
            got = _fetch_levels()
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            print(f"  [FETCH WARN] 시도 {attempts}/{FETCH_ATTEMPTS}: {err}")
            continue
        # 시도끼리 칸 단위 합집합 — 한 응답의 누락을 다른 응답이 메움
        fresh = got if fresh is None else fresh.combine_first(got)
        # 열 전체가 빈 응답도 잡도록 LKG 를 깐 상태에서 새 행을 본다
        gaps = _suspect_gaps(fresh if lkg is None else fresh.combine_first(lkg), lkg_last)
        if not gaps:
            break
        print(f"  [FETCH] 시도 {attempts}: 새 행에 누락 의심 {gaps} — 재시도")

    fresh_ok = fresh is not None and len(fresh) > 0
    fresh_last = fresh.index.max() if fresh_ok else None
    if lkg is None:
        if not fresh_ok or len(fresh) < MIN_LEVEL_ROWS:
            raise RuntimeError(f"bootstrap 불가 — LKG 없음 + fetch 결과 부족 ({err})")
        levels, mode = fresh, "bootstrap"
    elif not fresh_ok:
        levels, mode = lkg, "lkg_only"
    elif fresh_last > lkg_last:
        levels, mode = fresh.combine_first(lkg), "advanced"
    else:
        levels, mode = lkg.combine_first(fresh), "pinned"
    # combine_first 는 열을 합집합 정렬한다 — 기존 열 순서(섹터→SPY→매크로)로 되돌림
    cols = list(lkg.columns) if lkg is not None else list(fresh.columns)
    if fresh_ok:
        cols += [c for c in fresh.columns if c not in cols]
    levels = levels[cols]

    def _cells(df):
        return int(df.notna().to_numpy().sum()) if df is not None else 0

    report = {
        "generated":   pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "mode":        mode,
        "attempts":    attempts,
        "fetch_error": err,
        "lkg_last":    lkg_last.date().isoformat() if lkg_last is not None else None,
        "fresh_last":  fresh_last.date().isoformat() if fresh_last is not None else None,
        "levels_last": levels.index.max().date().isoformat(),
        "rows":        len(levels),
        # advanced: fresh 의 빈 칸을 LKG 가 메운 수 / pinned: fresh 가 LKG 빈 칸을 메운 수
        "patched_from_lkg": (_cells(levels) - _cells(fresh)) if mode == "advanced" else 0,
        "filled_into_lkg":  (_cells(levels) - _cells(lkg)) if mode == "pinned" else 0,
        "suspect_gaps": _suspect_gaps(levels, lkg_last) if mode in ("advanced", "bootstrap") else [],
        "nan_recent":  {c: [d.date().isoformat()
                            for d in levels.index[-10:][levels[c].iloc[-10:].isna().to_numpy()]]
                        for c in levels.columns if levels[c].iloc[-10:].isna().any()},
    }
    if mode in ("bootstrap", "advanced") or report["filled_into_lkg"] > 0:
        levels.to_csv(LKG_LEVELS_CSV)
    return levels, report


def load_all_stable() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """라이브(CI) 경로 — LKG 로 안정화한 load_all. 반환 마지막 원소 = 데이터 품질 리포트."""
    levels, report = _stable_levels()
    px_cols = [c for c in levels.columns if c not in MACRO_TICKERS]
    mc_cols = [c for c in levels.columns if c in MACRO_TICKERS]
    return (*_assemble(levels[px_cols], levels[mc_cols]), report)


def _publish_report(report: dict) -> None:
    DATA_QUALITY_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    notable = (report["mode"] in ("lkg_only", "bootstrap") or report["patched_from_lkg"]
               or report["filled_into_lkg"] or report["suspect_gaps"])
    line = (f"- 데이터 계층: **{report['mode']}** (levels_last={report['levels_last']}, "
            f"LKG 메움 {report['patched_from_lkg']} / LKG 보강 {report['filled_into_lkg']}, "
            f"누락 의심 {report['suspect_gaps'] or '없음'}, 시도 {report['attempts']})")
    print(line)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary and notable:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ret, mchg, px, mlvl, report = load_all_stable()
    _publish_report(report)
    ret.to_parquet(OUTPUT_DIR / "sector_returns.parquet")
    mchg.to_parquet(OUTPUT_DIR / "macro_changes.parquet")
    px.to_parquet(OUTPUT_DIR / "sector_prices.parquet")
    mlvl.to_parquet(OUTPUT_DIR / "macro_levels.parquet")
    print(f"\nSaved. Returns: {ret.shape}  Macro changes: {mchg.shape}")
    print(ret.tail(3))
    print(mchg.tail(3))

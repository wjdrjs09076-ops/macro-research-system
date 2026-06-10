"""
garch_estimator.py — 섹터별 GARCH(1,1) 조건부 변동성 및 변동성 위험 프리미엄(VRP) 추정

Black-Scholes로 내재변동성(IV)을 직접 추출하려면 옵션 체인 데이터가 필요하다.
구독 데이터에 옵션 데이터가 없으므로 GARCH(1,1)의 조건부 표준편차를 IV 대리변수로 사용한다.

  GARCH 조건부 변동성 (σ_GARCH) = 시장이 내일 기대하는 변동성의 모델 추정치
  실현 변동성 (RV)               = 과거 20일 수익률의 연율화 표준편차
  변동성 위험 프리미엄 (VRP)     = σ_GARCH - RV

VRP > 0  : 시장이 실현 변동성보다 높은 변동성을 기대 → 방어/헤지 프리미엄 상태
VRP < 0  : 시장이 실현 변동성보다 낮게 기대 → 과신 상태 (tail risk 경고)
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from arch import arch_model

from config import RV_WINDOW, OUTPUT_DIR


# ─────────────────────────────────────────────────────────────────────────────
# 단일 시리즈 GARCH 적합
# ─────────────────────────────────────────────────────────────────────────────

def fit_garch_single(returns_series: pd.Series) -> pd.Series:
    """
    GARCH(1,1) 적합 → 연율화 조건부 변동성 (소수점, e.g. 0.20 = 20%).

    입력 수익률은 로그 수익률(소수점 단위)을 기대한다.
    수치 안정성을 위해 내부적으로 ×100 스케일링 후 역변환한다.
    """
    s = returns_series.dropna()
    if len(s) < 100:
        return pd.Series(np.nan, index=returns_series.index, name=returns_series.name)

    r_pct = s * 100  # % 단위로 스케일

    try:
        am  = arch_model(r_pct, vol="Garch", p=1, q=1, dist="normal", rescale=False)
        res = am.fit(disp="off", show_warning=False, options={"ftol": 1e-9, "maxiter": 500})

        # conditional_volatility: 일간 % 단위 → 연율화 소수점 단위
        cond_vol_annual = res.conditional_volatility * np.sqrt(252) / 100
        return pd.Series(cond_vol_annual.values, index=s.index, name=returns_series.name)

    except Exception as e:
        print(f"    [GARCH FAIL] {returns_series.name}: {e}")
        return pd.Series(np.nan, index=returns_series.index, name=returns_series.name)


# ─────────────────────────────────────────────────────────────────────────────
# 전체 섹터 배치 처리
# ─────────────────────────────────────────────────────────────────────────────

def compute_garch_vol(returns: pd.DataFrame) -> pd.DataFrame:
    """모든 섹터(SPY 제외)에 GARCH(1,1) 적합. 결과: 연율화 조건부 변동성 DataFrame."""
    tickers = [c for c in returns.columns if c != "SPY"]
    results = {}
    for tkr in tickers:
        print(f"  GARCH({tkr})...")
        results[tkr] = fit_garch_single(returns[tkr])
    return pd.DataFrame(results)


def compute_realized_vol(returns: pd.DataFrame, window: int = RV_WINDOW) -> pd.DataFrame:
    """롤링 실현 변동성 (연율화, 소수점). window 거래일 기준."""
    return returns.rolling(window, min_periods=window // 2).std() * np.sqrt(252)


def compute_vrp(garch_vol: pd.DataFrame, realized_vol: pd.DataFrame) -> pd.DataFrame:
    """
    VRP = GARCH 조건부 변동성 - 실현 변동성 (두 변수 모두 연율화 소수점 단위).
    공통 컬럼·날짜 기준으로 정렬.
    """
    cols = garch_vol.columns.intersection(realized_vol.columns)
    idx  = garch_vol.index.intersection(realized_vol.index)
    return garch_vol.loc[idx, cols] - realized_vol.loc[idx, cols]


# ─────────────────────────────────────────────────────────────────────────────
# 독립 실행
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    returns = pd.read_parquet(OUTPUT_DIR / "sector_returns.parquet")

    print("Computing realized volatility...")
    rv = compute_realized_vol(returns)

    print("Fitting GARCH(1,1) for each sector...")
    gv = compute_garch_vol(returns)

    vrp = compute_vrp(gv, rv)

    rv.to_parquet(OUTPUT_DIR / "realized_vol.parquet")
    gv.to_parquet(OUTPUT_DIR / "garch_vol.parquet")
    vrp.to_parquet(OUTPUT_DIR / "vrp.parquet")
    vrp.to_csv(OUTPUT_DIR / "vrp_series.csv")

    print("\nLatest VRP snapshot:")
    snap = vrp.dropna(how="all").tail(1).T
    snap.columns = ["VRP (annualized)"]
    print(snap.round(4))

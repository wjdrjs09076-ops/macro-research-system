"""
mvg_mc.py - 섹터 다변량 가우시안 MC + NORTA + 콜레스키 분산귀인.

기여
─────
1) 섹터 공분산 Σ → 가우시안 콜레스키 분해 (변수 순서: |corr| 가중
   eigenvector centrality 내림차순 = '시스템 충격원' 우선).
2) 분산 분해(variance attribution): 각 섹터 분산이 어느 섹터의 *직교* 충격에서
   왔는지 — var_attr[i][j] = L[i,j]² / Σ_k L[i,k]² (행합=1).
3) NORTA 샘플 함수: 가우시안 공간에서 상관 주입 후 마진을 GPD-mixture로 변환
   (옵션). 현재는 정규 마진만 — GPD 합성은 후속.

설계 메모
──────────
- **인과 아님 — 가정의 명시**: 본 모듈이 산출하는 var_attr 와 propagation_score 는
  콜레스키 변수 순서에 100% 조건부. PCMCI 인과 체인은 매크로 변수 사이에서만
  발견 가능(월 단위) → 섹터 cross-section 엔 미적용. 섹터 순서는 |corr| 가중
  eigenvector centrality 라는 *임의 휴리스틱*으로 결정하며 (deterministic, fallback=
  알파벳 순), 따라서 "centrality 1위가 시스템 충격원" 은 *prior* 이지 발견이 아님.
- 꼬리 의존성(λ_L) 은 코퓰러 레이어(CO_CRASH_WITH) 가 별도로 캡처. 본 모듈은
  *평상시 가우시안 공동 변동* 의 상관 분해에 집중. CO_MOVES_WITH 엣지가 그 출력.
- 콜레스키는 본질적으로 순서 의존적 — 다른 ordering 으로 분해하면 var_attr 달라짐.
  이는 SVAR 재귀식별의 알려진 성질이며 본 모듈은 그 한계를 그대로 상속한다.

캐시 입력 : output/sector_returns.parquet
산출 객체 : CholeskyDecomp(order, corr, chol, var_attr, centrality)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import OUTPUT_DIR, SECTOR_ETFS

SECTOR_TICKERS: list[str] = list(SECTOR_ETFS.keys())
DEFAULT_WINDOW: int = 252


@dataclass
class CholeskyDecomp:
    """콜레스키 분해 + 분산 귀인 결과."""
    order: list[str]                 # centrality 내림차순 정렬
    corr: pd.DataFrame               # 정렬된 상관행렬 (P × P)
    chol: np.ndarray                 # 하삼각 L: L L^T = corr
    var_attr: pd.DataFrame           # row=섹터(target), col=섹터(source shock), 행합=1
    centrality: dict[str, float]     # ordering 점수

    def shock_contribution_to(self, target: str) -> dict[str, float]:
        """target 분산을 만든 직교 충격들의 비중 (자기 자신 포함, 합=1)."""
        return self.var_attr.loc[target].to_dict()

    def shocks_caused_by(self, source: str) -> dict[str, float]:
        """source의 직교 충격이 *다른* 섹터들의 분산에 기여한 비중."""
        return self.var_attr[source].to_dict()

    def propagation_score(self, source: str) -> float:
        """source 의 직교 충격이 다른 섹터 분산에 끼친 총 기여 (자기 제외).

        높을수록 '시스템 전파자'.
        """
        col = self.var_attr[source]
        return float(col.sum() - col.loc[source])

    def self_share(self, target: str) -> float:
        """target 분산 중 자기 자신 직교 충격이 차지하는 비중. 1에 가까울수록 '독립'."""
        return float(self.var_attr.loc[target, target])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_returns() -> pd.DataFrame:
    path = OUTPUT_DIR / "sector_returns.parquet"
    if not path.exists():
        raise FileNotFoundError(f"sector_returns.parquet not found at {path}")
    df = pd.read_parquet(path)
    cols = [c for c in SECTOR_TICKERS if c in df.columns]
    return df[cols].dropna()


def _eigenvector_centrality_order(
    corr: pd.DataFrame,
) -> tuple[list[str], dict[str, float]]:
    """|corr| 가중 무방향 그래프의 eigenvector centrality 내림차순.

    실패 시 (alphabet, NaN) 폴백.
    """
    try:
        G = nx.from_pandas_adjacency(corr.abs())
        G.remove_edges_from(nx.selfloop_edges(G))
        c = nx.eigenvector_centrality_numpy(G, weight="weight")
        order = sorted(c.keys(), key=lambda k: -c[k])
        return order, {k: float(v) for k, v in c.items()}
    except Exception as exc:  # pragma: no cover
        print(f"  [WARN] eigenvector centrality failed ({exc}); falling back to alphabetic")
        cols = sorted(corr.columns)
        return cols, {k: float("nan") for k in cols}


def _spd_safe(M: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """대칭화 + 최소 고유값 ridge 보정 (Cholesky 수치 안정)."""
    M = (M + M.T) / 2
    w = np.linalg.eigvalsh(M)
    if w.min() <= eps:
        M = M + np.eye(M.shape[0]) * (eps - w.min())
    return M


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cholesky_decompose(window: int = DEFAULT_WINDOW) -> CholeskyDecomp:
    """최근 `window`일 섹터 수익률 → 콜레스키 분해 + 분산 귀인.

    Cholesky 변수 순서 = |corr| 가중 그래프의 eigenvector centrality 내림차순.

    Notes
    -----
    Σ = LL^T (정렬된 상관행렬). 정규성 가정 하에서
        Var(r_i) = Σ_k L[i,k]² × Var(z_k),  z_k 독립 N(0,1)
    이므로 var_attr[i,j] = L[i,j]² / Σ_k L[i,k]² 는 '직교 충격 j 가 r_i 분산에
    기여한 비율' (행합=1).
    """
    rets = _load_returns().tail(window)
    if len(rets) < 30:
        raise ValueError(f"not enough returns for cholesky: {len(rets)}")

    corr = rets.corr()
    order, centrality = _eigenvector_centrality_order(corr)
    corr = corr.loc[order, order]

    L = np.linalg.cholesky(_spd_safe(corr.values))

    sq = L ** 2
    row_sum = sq.sum(axis=1, keepdims=True)
    attr = sq / np.where(row_sum > 0, row_sum, 1.0)
    var_attr = pd.DataFrame(attr, index=order, columns=order)

    return CholeskyDecomp(
        order=order,
        corr=corr,
        chol=L,
        var_attr=var_attr,
        centrality=centrality,
    )


def shapley_decompose(
    window: int = DEFAULT_WINDOW,
    n_samples: int = 200,
    seed: int = 42,
) -> CholeskyDecomp:
    """Shapley-Owen 분산 분해 — Castro et al. (2009) sampling Shapley.

    *순서 독립적* 변수 기여 분해. 모든 P! ordering 의 var_attr 평균.
    11! = 4천만 → M=n_samples 무작위 ordering 추출해서 평균 (수렴 빠름, M=200 충분).

    **인과 의미는 여전히 없음** (라운드 4 비평 반영): Shapley 는 *공정 배분* 의 비인과
    공식. ordering 의존성만 제거하고 "centrality 1위 = 충격원" 라는 임의 휴리스틱을
    제거한다. propagation_score / self_share 의 알고리즘 정직성 향상.

    Returns
    -------
    CholeskyDecomp (호환성 위해 같은 dataclass 사용):
      order        : self_share 내림차순 (= 가장 idiosyncratic 한 섹터부터)
      corr         : 정렬되지 않은 원본 상관행렬
      chol         : NaN (정의 안 됨 — Shapley 는 단일 L 없음)
      var_attr     : Shapley 평균 분해 (행합=1, 인덱스 = 원본 ticker 순)
      centrality   : propagation_score (Shapley 기준) 로 의미 재정의
    """
    rets = _load_returns().tail(window)
    if len(rets) < 30:
        raise ValueError(f"not enough returns for shapley: {len(rets)}")

    corr_df = rets.corr()
    tickers = list(corr_df.columns)
    P = len(tickers)
    Sigma_full = _spd_safe(corr_df.values)

    rng = np.random.default_rng(seed)
    # M 개 무작위 순서 × 각 ordering 에서 Cholesky 분해 → ticker-space 로 매핑해서 합산
    sum_attr = np.zeros((P, P), dtype=float)
    n_eff = 0
    for _ in range(n_samples):
        perm = rng.permutation(P)
        Sigma_perm = Sigma_full[np.ix_(perm, perm)]
        try:
            L = np.linalg.cholesky(Sigma_perm)
        except np.linalg.LinAlgError:
            continue
        sq = L ** 2
        row_sum = sq.sum(axis=1, keepdims=True)
        attr_perm = sq / np.where(row_sum > 0, row_sum, 1.0)
        # perm-space → ticker-space 로 역매핑
        attr_full = np.zeros((P, P), dtype=float)
        for ii, i_orig in enumerate(perm):
            for jj, j_orig in enumerate(perm):
                attr_full[i_orig, j_orig] = attr_perm[ii, jj]
        sum_attr += attr_full
        n_eff += 1
    if n_eff == 0:
        raise RuntimeError("Shapley sampling: all orderings produced cholesky failures")
    var_attr_arr = sum_attr / n_eff

    var_attr = pd.DataFrame(var_attr_arr, index=tickers, columns=tickers)

    # self_share 내림차순으로 order 부여 (centrality 휴리스틱 대체)
    self_share = {t: float(var_attr.loc[t, t]) for t in tickers}
    order = sorted(tickers, key=lambda t: -self_share[t])
    var_attr = var_attr.loc[order, order]

    # propagation_score = Σ_{i≠j} var_attr[i, j] 으로 centrality 채움
    propagation = {
        t: float(var_attr[t].sum() - var_attr.loc[t, t]) for t in order
    }

    return CholeskyDecomp(
        order=order,
        corr=corr_df.loc[order, order],
        chol=np.full((P, P), np.nan),   # Shapley 는 단일 L 없음
        var_attr=var_attr,
        centrality=propagation,
    )


def sample_norta(
    decomp: CholeskyDecomp,
    n_sim: int,
    horizon: int,
    inverse_marginal: dict[str, "callable"] | None = None,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """NORTA 다변량 샘플: 가우시안 공간 상관 → 마진 정규 → (옵션) GPD 합성 역변환.

    inverse_marginal 이 주어지면 각 섹터의 (정규공간 → 실수익률) 역CDF 적용
    (예: GPD 꼬리 + 경험분포 본체 합성). 미지정 시 정규 마진 그대로.
    """
    rng = np.random.default_rng(seed)
    P = len(decomp.order)
    z = rng.standard_normal((n_sim, horizon, P))
    g = z @ decomp.chol.T  # 가우시안 공간 상관 주입 (P 축이 마지막)

    out: dict[str, np.ndarray] = {}
    if inverse_marginal:
        from scipy.stats import norm  # type: ignore
        u = norm.cdf(g)
        for j, t in enumerate(decomp.order):
            fn = inverse_marginal.get(t)
            out[t] = fn(u[:, :, j]) if fn else g[:, :, j]
    else:
        for j, t in enumerate(decomp.order):
            out[t] = g[:, :, j]
    return out


# ---------------------------------------------------------------------------
# Joint path simulation + drawdown stats
# ---------------------------------------------------------------------------

def _load_latest_garch_vol() -> dict[str, float] | None:
    """garch_vol.parquet 최신 행 → 섹터별 *연환산* 변동성 dict. 없으면 None."""
    path = OUTPUT_DIR / "garch_vol.parquet"
    if not path.exists():
        return None
    v = pd.read_parquet(path).iloc[-1]
    return {k: float(v[k]) for k in v.index if pd.notna(v[k])}


def simulate_joint_paths(
    decomp: CholeskyDecomp,
    sigmas_annual: dict[str, float],
    n_sim: int = 5000,
    horizon: int = 21,
    drift_annual: float = 0.0,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """다변량 GBM 경로 시뮬: NORTA(가우시안 상관) + 섹터별 σ_daily + 드리프트.

    반환: ticker → (n_sim, horizon) 로그 *누적* 수익률 배열.
      cum[i, h] = Σ_{t<=h} r_t  ⇒ drawdown = min over h
    """
    rng = np.random.default_rng(seed)
    P = len(decomp.order)
    sigma_d = np.array([
        sigmas_annual.get(t, 0.18) / np.sqrt(252.0) for t in decomp.order
    ])  # 누락 섹터는 18% 연환산 폴백
    mu_d = drift_annual / 252.0

    z = rng.standard_normal((n_sim, horizon, P))
    g = z @ decomp.chol.T                       # 상관 주입 표준정규
    r = mu_d + g * sigma_d[None, None, :]       # 일별 로그수익률
    cum = r.cumsum(axis=1)                      # 누적

    return {t: cum[:, :, j] for j, t in enumerate(decomp.order)}


def joint_drawdown_stats(
    paths: dict[str, np.ndarray],
    thresholds: list[float] | None = None,
) -> dict:
    """공동 drawdown 통계 — '시그널이 옳다는 가정 없이 시장 구조만으로 본 위험도'.

    각 path에서 섹터별 최대 손실 = min over horizon of cumulative log return.
    임계값 τ 에 대해:
      p_any   = P(적어도 한 섹터 ≤ -τ)
      p_all   = P(모든 섹터 ≤ -τ)         ← 시스템 동시 폭락
      p_half  = P(≥절반 섹터 ≤ -τ)
      mean_hit_count = 평균 동시 hit 섹터 수
    페어 공동 drawdown(5% 임계값 기준 상위 10개도 노출).
    """
    if thresholds is None:
        thresholds = [0.03, 0.05, 0.10]

    tickers = list(paths.keys())
    dd = np.stack([paths[t].min(axis=1) for t in tickers], axis=1)  # (n_sim, P)
    P = dd.shape[1]
    half = (P + 1) // 2

    per_threshold: list[dict] = []
    for tau in thresholds:
        hit = dd <= -tau
        n_hit = hit.sum(axis=1)
        per_threshold.append({
            "threshold":      round(float(tau), 4),
            "p_any":          round(float((n_hit >= 1).mean()), 4),
            "p_all":          round(float((n_hit == P).mean()), 4),
            "p_half":         round(float((n_hit >= half).mean()), 4),
            "mean_hit_count": round(float(n_hit.mean()), 2),
        })

    base_tau = 0.05
    hit = dd <= -base_tau
    pair_hits: list[dict] = []
    for i in range(P):
        for j in range(i + 1, P):
            p = float((hit[:, i] & hit[:, j]).mean())
            pair_hits.append({"a": tickers[i], "b": tickers[j], "p_co": round(p, 4)})
    pair_hits.sort(key=lambda x: -x["p_co"])

    return {
        "tickers":              tickers,
        "horizon":              int(next(iter(paths.values())).shape[1]),
        "n_sim":                int(next(iter(paths.values())).shape[0]),
        "per_threshold":        per_threshold,
        "pair_threshold":       base_tau,
        "top_pair_co_drawdown": pair_hits[:10],
    }


def run_joint_simulation(
    n_sim: int = 5000,
    horizon: int = 21,
    seed: int = 42,
) -> dict:
    """편의 함수: cholesky_decompose() + GARCH 변동성 로드 + 시뮬 + 통계."""
    d = cholesky_decompose()
    sigmas = _load_latest_garch_vol() or {}
    paths = simulate_joint_paths(d, sigmas, n_sim=n_sim, horizon=horizon, seed=seed)
    stats = joint_drawdown_stats(paths)
    stats["sigmas_annual"] = {t: round(sigmas.get(t, 0.18), 4) for t in d.order}
    return stats


# ---------------------------------------------------------------------------
# CLI diagnostic
# ---------------------------------------------------------------------------

def _print_top_propagators(d: CholeskyDecomp, k: int = 5) -> None:
    scores = sorted(
        ((t, d.propagation_score(t)) for t in d.order),
        key=lambda kv: -kv[1],
    )
    print(f"\n  Top {k} shock propagators (총 외부 기여):")
    for t, s in scores[:k]:
        print(f"    {t:5s}  prop={s:.3f}  centrality={d.centrality.get(t, 0):.3f}  "
              f"self={d.self_share(t):.3f}")


def _print_top_concentrated(d: CholeskyDecomp, k: int = 5) -> None:
    rows = sorted(((t, d.self_share(t)) for t in d.order), key=lambda kv: -kv[1])
    print(f"\n  Top {k} variance-concentrated (자기 충격 비중):")
    for t, s in rows[:k]:
        print(f"    {t:5s}  self_share={s:.3f}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print(f"[mvg_mc] sector_returns.parquet 로딩 → 최근 {DEFAULT_WINDOW}일")
    d = cholesky_decompose()
    print(f"  변수 순서 (centrality 내림차순): {' → '.join(d.order)}")
    print(f"  상관행렬 평균 = {d.corr.values.mean():.3f}, "
          f"최대 비대각 = {(d.corr.values - np.eye(len(d.order))).max():.3f}")

    _print_top_propagators(d)
    _print_top_concentrated(d)

    print(f"\n[joint sim] horizon=21, n_sim=5000 ...")
    stats = run_joint_simulation()
    print(f"  공동 drawdown 확률 (21영업일 horizon):")
    for row in stats["per_threshold"]:
        print(f"    τ={row['threshold']*100:>4.1f}%  "
              f"p_any={row['p_any']:.3f}  p_half={row['p_half']:.3f}  "
              f"p_all={row['p_all']:.3f}  hit_avg={row['mean_hit_count']:.2f}")
    print(f"\n  상위 5 페어 공동 drawdown (τ={stats['pair_threshold']*100:.0f}%):")
    for pair in stats["top_pair_co_drawdown"][:5]:
        print(f"    {pair['a']:5s}-{pair['b']:<5s}  p_co={pair['p_co']:.3f}")


if __name__ == "__main__":
    main()

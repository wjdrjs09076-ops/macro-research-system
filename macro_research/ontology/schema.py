"""
schema.py - Ontology schema: node/edge type definitions + NetworkX DiGraph structure

Object Layer  : Sector, MacroFactor, Regime
Link Layer    : SENSITIVE_TO, CO_CRASH_WITH, HAS_TAIL_RISK, HAS_VRP
Action Layer  : Inference rules produce Signal nodes with reasoning chains
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import networkx as nx


# ---------------------------------------------------------------------------
# Enum types
# ---------------------------------------------------------------------------

class NodeType(str, Enum):
    SECTOR       = "Sector"
    MACRO_FACTOR = "MacroFactor"
    REGIME       = "Regime"
    SIGNAL       = "Signal"


class EdgeType(str, Enum):
    SENSITIVE_TO   = "SENSITIVE_TO"      # sector -> macro: delta/gamma/speed/color
    CO_CRASH_WITH  = "CO_CRASH_WITH"     # sector -> sector: lower tail dependence λ_L
    CO_MOVES_WITH  = "CO_MOVES_WITH"     # sector -> sector: 평상시 가우시안 의존성 (Cholesky 분산기여)
    HAS_TAIL_RISK  = "HAS_TAIL_RISK"     # sector -> sector: GPD params, CF VaR
    HAS_VRP        = "HAS_VRP"           # sector -> signal source
    REGIME_ACTIVE  = "REGIME_ACTIVE"     # regime -> (contextual)
    GENERATES      = "GENERATES"         # sector -> signal (from inference)
    TRANSMITS_TO   = "TRANSMITS_TO"      # macro -> macro: PCMCI causal link (lag, stability)


class SignalType(str, Enum):
    OVERWEIGHT   = "OVERWEIGHT"
    UNDERWEIGHT  = "UNDERWEIGHT"
    HEDGE        = "HEDGE"
    MONITOR      = "MONITOR"


class RegimeType(str, Enum):
    LOW_VIX  = "low_vix"   # VIX < 15
    MID_VIX  = "mid_vix"   # 15 <= VIX < 25
    HIGH_VIX = "high_vix"  # VIX >= 25


# ---------------------------------------------------------------------------
# Node dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SectorNode:
    ticker: str
    name: str
    node_type: NodeType = NodeType.SECTOR

    # Tail risk attributes (populated by populate.py)
    xi: float = float("nan")
    var99: float = float("nan")
    es99: float = float("nan")
    cf_var99: float = float("nan")
    cf_over_normal: float = float("nan")
    skewness: float = float("nan")
    kurtosis: float = float("nan")

    # VRP attributes
    vrp_latest: float = float("nan")   # 구 GARCH 기반 (σ_GARCH - RV) — 진짜 VRP 아님
    vrp_iv: float = float("nan")        # 진짜 VRP = ATM_IV - RV_20d (Q vs P 갭, 2026-06-08~)
    iv_atm: float = float("nan")        # 섹터 ATM 콜+풋 평균 IV (연환산)

    # Cholesky variance decomposition — centrality 순서 조건부, 인과 아님
    # var_decomposition[j] = 임의 순서 콜레스키의 재귀 잔차 행 분해 (합=1)
    var_decomposition: dict[str, float] = field(default_factory=dict)
    self_share: float = float("nan")        # var_decomposition[self], 순서 조건부
    propagation_score: float = float("nan") # 순서 1위일 때 다른 행의 첫 컬럼 합
    cholesky_order: int = -1                # 0=centrality 1위 (가정), 인과 아님

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_type":         self.node_type.value,
            "name":              self.name,
            "xi":                self.xi,
            "var99":             self.var99,
            "es99":              self.es99,
            "cf_var99":          self.cf_var99,
            "cf_over_normal":    self.cf_over_normal,
            "skewness":          self.skewness,
            "kurtosis":          self.kurtosis,
            "vrp_latest":        self.vrp_latest,
            "vrp_iv":            self.vrp_iv,
            "iv_atm":            self.iv_atm,
            "var_decomposition": dict(self.var_decomposition),
            "self_share":        self.self_share,
            "propagation_score": self.propagation_score,
            "cholesky_order":    self.cholesky_order,
        }


@dataclass
class MacroFactorNode:
    ticker: str
    name: str
    node_type: NodeType = NodeType.MACRO_FACTOR

    def to_dict(self) -> dict[str, Any]:
        return {"node_type": self.node_type.value, "name": self.name}


@dataclass
class RegimeNode:
    regime_id: str
    label: str
    vix_range: tuple[float, float]
    node_type: NodeType = NodeType.REGIME

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_type": self.node_type.value,
            "label":     self.label,
            "vix_lo":    self.vix_range[0],
            "vix_hi":    self.vix_range[1],
        }


@dataclass
class SignalNode:
    signal_id: str
    sector: str
    signal_type: SignalType
    regime: str
    confidence: float
    rule_name: str
    reasoning: list[str] = field(default_factory=list)
    node_type: NodeType = NodeType.SIGNAL

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_type":   self.node_type.value,
            "sector":      self.sector,
            "signal_type": self.signal_type.value,
            "regime":      self.regime,
            "confidence":  self.confidence,
            "rule_name":   self.rule_name,
            "reasoning":   " | ".join(self.reasoning),
        }


# ---------------------------------------------------------------------------
# Edge dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SensitiveToEdge:
    """Sector -> MacroFactor sensitivity from OLS / poly4 regression.

    delta/gamma          : 이변량 2차 OLS (매크로 1개씩) — 교란 노출
    delta_ctrl/gamma_ctrl: 다변량 partial (전 매크로 동시 + SPY 통제, 2026-06-12)
                           rate_* 룰은 ctrl 우선 사용 (유가↑→금리↑ 교란 제거)
    """
    delta: float = float("nan")
    gamma: float = float("nan")
    speed: float = float("nan")
    color: float = float("nan")
    t_delta: float = float("nan")
    t_gamma: float = float("nan")
    r2: float = float("nan")
    tail_asymmetry: float = float("nan")    # beta(0.05) - beta(0.95)
    tail_dep_lower: float = float("nan")    # lambda_L sector->macro
    delta_ctrl: float = float("nan")        # partial delta (타 매크로+SPY 통제)
    gamma_ctrl: float = float("nan")
    t_delta_ctrl: float = float("nan")
    beta_mkt: float = float("nan")          # 시장(SPY) 통제 베타
    vif: float = float("nan")               # 해당 매크로 regressor 의 VIF (공선성 진단)

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_type":      EdgeType.SENSITIVE_TO.value,
            "delta":          self.delta,
            "gamma":          self.gamma,
            "speed":          self.speed,
            "color":          self.color,
            "t_delta":        self.t_delta,
            "t_gamma":        self.t_gamma,
            "r2":             self.r2,
            "tail_asymmetry": self.tail_asymmetry,
            "tail_dep_lower": self.tail_dep_lower,
            "delta_ctrl":     self.delta_ctrl,
            "gamma_ctrl":     self.gamma_ctrl,
            "t_delta_ctrl":   self.t_delta_ctrl,
            "beta_mkt":       self.beta_mkt,
            "vif":            self.vif,
        }


@dataclass
class CoCrashEdge:
    """Sector -> Sector lower tail dependence.

    lambda_lower_hi (2026-06-12): 블록 부트스트랩 95% 상한. 점추정 λ_L 은
    u=0.05 조건부 표본 ~18개라 SE>0.1 — 무한손실 방향(thin_tail_greenlight)
    판정은 상한으로 한다.
    """
    lambda_lower: float = float("nan")
    lambda_upper: float = float("nan")
    lambda_lower_hi: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_type":       EdgeType.CO_CRASH_WITH.value,
            "lambda_lower":    self.lambda_lower,
            "lambda_upper":    self.lambda_upper,
            "lambda_lower_hi": self.lambda_lower_hi,
        }


@dataclass
class CoMovesEdge:
    """Sector -> Sector 평상시 가우시안 공동 변동 (Cholesky 잔차 분해, 순서 조건부).

    *인과 아님*: source → target 의 화살표 방향은 |corr| centrality 순서로 정한
    가정에 100% 의존. 다른 순서로 분해하면 attribution 도 바뀜.
      attribution : L[target, source]² / Σ_k L[target, k]² ∈ [0, 1] (순서 조건부)
      pearson_corr: 정렬 전 단순 Pearson 상관 (대칭값)
    CO_CRASH_WITH(꼬리, 대칭) 와 직교 — 본체 의존성 vs 꼬리 의존성 분업.
    """
    attribution: float = float("nan")
    pearson_corr: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_type":    EdgeType.CO_MOVES_WITH.value,
            "attribution":  self.attribution,
            "pearson_corr": self.pearson_corr,
        }


@dataclass
class TransmitsToEdge:
    """MacroFactor -> MacroFactor causal link discovered by PCMCI."""
    lag_months: int   = 0
    coef:       float = float("nan")
    p_value:    float = float("nan")
    stability:  str   = "historical"   # structural / emerging / historical
    score:      float = float("nan")
    direction:  str   = "+"            # "+" positive / "-" negative causal effect
    best_lag:   int   = 0              # lag with lowest p-value for this pair

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_type":  EdgeType.TRANSMITS_TO.value,
            "lag_months": self.lag_months,
            "coef":       self.coef,
            "p_value":    self.p_value,
            "stability":  self.stability,
            "score":      self.score,
            "direction":  self.direction,
        }


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

SECTOR_NAMES: dict[str, str] = {
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

# OIL 은 2026-06-12 부터 일별 SENSITIVE_TO 대상에도 포함 (CAUSAL_MACRO_NAMES
# 의 월별 PCMCI 노드와 동일 노드 — build_empty_graph 에서 중복 add 는 무해).
MACRO_NAMES: dict[str, str] = {
    "VIX":   "CBOE Volatility Index",
    "US10Y": "US 10Y Treasury Yield",
    "US2Y":  "US 2Y Treasury Yield",
    "OIL":   "WTI Crude Oil Price",
    "DXY":   "US Dollar Index",
}

# PCMCI 인과 발견에서 추가된 매크로 변수 (TRANSMITS_TO 엣지 전용)
CAUSAL_MACRO_NAMES: dict[str, str] = {
    "OIL":           "WTI Crude Oil Price",
    "CPI":           "Consumer Price Index",
    "FED_FUNDS":     "Federal Funds Rate",
    "CREDIT_SPREAD": "Moody's Baa - 10Y Spread",
}

REGIME_DEFS: list[tuple[str, str, tuple[float, float]]] = [
    (RegimeType.LOW_VIX.value,  "Low Volatility (VIX < 15)",          (0.0,  15.0)),
    (RegimeType.MID_VIX.value,  "Medium Volatility (15 <= VIX < 25)", (15.0, 25.0)),
    (RegimeType.HIGH_VIX.value, "High Volatility (VIX >= 25)",        (25.0, 9999.0)),
]


def build_empty_graph() -> nx.DiGraph:
    """Return a DiGraph pre-populated with Sector, MacroFactor, Regime, and CausalMacro nodes."""
    G = nx.DiGraph()

    # Sector nodes
    for ticker, name in SECTOR_NAMES.items():
        node = SectorNode(ticker=ticker, name=name)
        G.add_node(ticker, **node.to_dict())

    # MacroFactor nodes (SENSITIVE_TO targets)
    for ticker, name in MACRO_NAMES.items():
        node = MacroFactorNode(ticker=ticker, name=name)
        G.add_node(ticker, **node.to_dict())

    # Causal MacroFactor nodes (TRANSMITS_TO network from PCMCI)
    for ticker, name in CAUSAL_MACRO_NAMES.items():
        node = MacroFactorNode(ticker=ticker, name=name)
        G.add_node(ticker, **node.to_dict())

    # Regime nodes
    for regime_id, label, vix_range in REGIME_DEFS:
        node = RegimeNode(regime_id=regime_id, label=label, vix_range=vix_range)
        G.add_node(regime_id, **node.to_dict())

    return G

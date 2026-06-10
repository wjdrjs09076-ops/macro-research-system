"""
trading_ontology.py

트레이딩 온톨로지 — Beta 체제전환 + VRP 신호 프레임워크

역사적 위기 연구(닷컴/GFC/COVID/호르무즈)에서 도출된 지식을
4개 레이어 인과 그래프로 표현:

  Layer 1: 이벤트 유형 분류
  Layer 2: 인과 메커니즘
  Layer 3: 신호 유효성 조건
  Layer 4: 매매 결정 + 실패모드

새 이벤트 입력 → 분류 → 신호 적용 가능성 판단 → 방향 결정

출력:
  output/ontology_trading.json
  output/figures/trading_ontology.png
"""

import json
import warnings
warnings.filterwarnings("ignore")

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

BASE_DIR    = Path(__file__).parent
OUTPUT_DIR  = BASE_DIR / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ===========================================================================
# LAYER 1 — 이벤트 유형 분류 체계
# ===========================================================================

EVENT_TYPES = {
    "SUPPLY_SHOCK": {
        "label"               : "외부 공급 충격",
        "en"                  : "External Supply Shock",
        "description"         : "지정학 이벤트가 상품 공급 경로 제한 → 해당 섹터 전체가 시장과 역방향 (beta < 0)",
        "framework_applicable": True,
        "direction"           : "LONG",
        "beta_thresh"         : -0.05,
        "examples"            : ["호르무즈 봉쇄 → XLE", "우크라이나 전쟁 → XLE", "OPEC 감산 → XLE"],
        "validated_oos_sharpe": 2.1,   # 연구 결과 평균
        "color"               : "#27ae60",
        "layer"               : 1,
    },
    "STRUCTURAL_COLLAPSE": {
        "label"               : "구조적 섹터 붕괴",
        "en"                  : "Structural Sector Collapse",
        "description"         : "해당 섹터가 위기 진원지, 시스템 리스크로 beta 급등 + VRP 음전환 → SHORT 유효",
        "framework_applicable": True,
        "direction"           : "SHORT",
        "beta_thresh"         : 1.5,
        "examples"            : ["GFC → XLF", "저축대부조합 위기 → 금융주", "S&L 사태"],
        "validated_oos_sharpe": 0.9,
        "color"               : "#c0392b",
        "layer"               : 1,
    },
    "VALUATION_CORRECTION": {
        "label"               : "밸류에이션 조정",
        "en"                  : "Valuation Correction",
        "description"         : "과대평가 섹터 수년간 조정. 베어마켓 랠리 반복 → 신호가 공황 피크에서 발생, 랠리에 손절",
        "framework_applicable": False,
        "direction"           : "NONE",
        "beta_thresh"         : None,
        "examples"            : ["닷컴버블 → XLK (SHORT 실패)", "중국 부동산", "암호화폐 사이클"],
        "validated_oos_sharpe": -0.4,  # 실제 실패 결과
        "color"               : "#e67e22",
        "layer"               : 1,
    },
    "SUBSECTOR_SPECIFIC": {
        "label"               : "하위섹터 특수",
        "en"                  : "Sub-Sector Specific",
        "description"         : "ETF 내 소수 종목만 수혜/피해. 섹터 ETF 전체 beta 반전 없음 → 신호 불발",
        "framework_applicable": False,
        "direction"           : "NONE",
        "beta_thresh"         : None,
        "examples"            : ["COVID → XLV (0건 신호)", "AI 붐 → XLK (NVDA만 폭등)"],
        "validated_oos_sharpe": 0.0,
        "color"               : "#7f8c8d",
        "layer"               : 1,
    },
}

# ===========================================================================
# LAYER 2 — 인과 메커니즘
# ===========================================================================

MECHANISMS = {
    "GEO_SUPPLY_CUT": {
        "label"      : "지정학적 공급 차단",
        "description": "전쟁/봉쇄/제재가 상품(석유/가스/반도체) 공급 경로를 물리적으로 제한",
        "layer"      : 2,
        "color"      : "#2980b9",
    },
    "COMMODITY_PRICE_SPIKE": {
        "label"      : "상품 가격 급등",
        "description": "공급 감소 → 상품 현물가 급등 → 생산자 수익 급증",
        "layer"      : 2,
        "color"      : "#2980b9",
    },
    "ETF_DECOUPLING": {
        "label"      : "섹터 ETF 시장 디커플링",
        "description": "섹터가 시장 방향과 독립적으로 움직이기 시작. 핵심 전제 조건",
        "layer"      : 2,
        "color"      : "#2980b9",
    },
    "BETA_REVERSAL": {
        "label"      : "Beta 부호 반전",
        "description": "30D 롤링 beta가 임계값을 넘어 양→음 (LONG) 또는 1 초과 급등 (SHORT) 전환",
        "layer"      : 2,
        "color"      : "#2980b9",
    },
    "CAPM_IV_UNDERESTIMATE": {
        "label"      : "CAPM IV 과소추정",
        "description": "Beta 부호 반전 시 CAPM_IV=sqrt(β²·VIX²+σ_idio²)가 실제 변동성보다 낮게 추정됨",
        "layer"      : 2,
        "color"      : "#2980b9",
    },
    "VRP_FLIP": {
        "label"      : "VRP 음전환",
        "description": "CAPM_IV < 실현변동성(RV) → 옵션시장이 공포를 과소평가 → 추가 이동 예고",
        "layer"      : 2,
        "color"      : "#2980b9",
    },
    "SYSTEMIC_CREDIT": {
        "label"      : "시스템 신용 경색",
        "description": "금융 섹터 내 연쇄 도산/담보 마진콜 → beta 급등 + VRP 음전환 (GFC 패턴)",
        "layer"      : 2,
        "color"      : "#2980b9",
    },
    "BEAR_RALLY": {
        "label"      : "베어마켓 랠리",
        "description": "하락 추세 중 단기 강한 반등(10~30%). 신호가 공황 피크(VRP<0)에서 발생 후 역방향 스퀴즈",
        "layer"      : 2,
        "color"      : "#e74c3c",
    },
}

# ===========================================================================
# LAYER 3 — 신호 유효성 조건 (3가지 필요조건)
# ===========================================================================

VALIDITY_CONDITIONS = {
    "FULL_ETF_DECOUPLING": {
        "label"       : "섹터 ETF 전체 디커플링",
        "description" : "소수 종목이 아닌 ETF 전체(가중평균 beta)가 반전해야 함",
        "how_to_check": "30D 롤링 beta가 연속 3일 이상 임계값 돌파",
        "failed_in"   : ["COVID/XLV: 전체 디커플링 없음 → 신호 0건"],
        "layer"       : 3,
        "color"       : "#8e44ad",
    },
    "STRUCTURAL_PERSISTENCE": {
        "label"       : "구조적 지속성 (>20 거래일)",
        "description" : "체제 전환이 수주~수개월 유지. 단기 공황(수일)이면 랠리 스퀴즈 발생",
        "how_to_check": "beta 반전 후 평균 체류 기간 > 20 거래일 확인",
        "failed_in"   : ["단기 지정학 긴장 (1주 이내 해소)"],
        "layer"       : 3,
        "color"       : "#8e44ad",
    },
    "STRUCTURAL_CAUSE": {
        "label"       : "구조적(펀더멘털) 원인",
        "description" : "물리적 공급 제약 또는 시스템 리스크. 밸류에이션 조정 사유 제외",
        "how_to_check": "이벤트 원인이 물리적/제도적 제약인지 (밸류에이션이면 FAIL)",
        "failed_in"   : ["닷컴버블: 밸류에이션 조정이 원인 → SHORT 타이밍 실패"],
        "layer"       : 3,
        "color"       : "#8e44ad",
    },
}

# ===========================================================================
# LAYER 4 — 매매 결정 + 실패 모드
# ===========================================================================

TRADE_DECISIONS = {
    "LONG_BENEFICIARY": {
        "label"      : "수혜 섹터 매수 (LONG)",
        "action"     : "OVERWEIGHT",
        "description": "섹터 ETF 및 개별 종목 매수. S2_vrp 단독 또는 S3_combo 신호 사용",
        "tc_note"    : "US 0.05%/회. KR 0.35%/회 → KR은 신호 발생 빈도 중요",
        "layer"      : 4,
        "color"      : "#27ae60",
    },
    "SHORT_VICTIM": {
        "label"      : "피해 섹터 매도 (SHORT)",
        "action"     : "UNDERWEIGHT",
        "description": "섹터 ETF 및 개별 종목 공매도. beta > 1.5 AND VRP < 0 연속 3일",
        "tc_note"    : "공매도 수수료 별도. 대차 가능 여부 확인 필요",
        "layer"      : 4,
        "color"      : "#c0392b",
    },
    "MONITOR": {
        "label"      : "관찰 유지 (MONITOR)",
        "action"     : "MONITOR",
        "description": "프레임워크 부분 충족. 신호는 발생하나 유효성 조건 미충족. 포지션 없이 관찰",
        "tc_note"    : "포지션 없음",
        "layer"      : 4,
        "color"      : "#f39c12",
    },
    "AVOID": {
        "label"      : "적용 불가 (AVOID)",
        "action"     : "NEUTRAL",
        "description": "이벤트 유형이 프레임워크 전제 조건 불충족. 매매 신호 없음",
        "tc_note"    : "포지션 없음",
        "layer"      : 4,
        "color"      : "#7f8c8d",
    },
}

FAILURE_MODES = {
    "BEAR_RALLY_SQUEEZE": {
        "label"      : "베어마켓 랠리 스퀴즈",
        "description": "VRP<0(공황 피크)에서 SHORT 진입 → 단기 반등에 손실. 닷컴 OOS Sharpe=-1.2",
        "affects"    : ["VALUATION_CORRECTION"],
        "mitigation" : "consec >= 5일 필터 또는 모멘텀 필터(20D MA 방향 확인) 추가",
        "layer"      : 4,
        "color"      : "#e74c3c",
    },
    "SUBSECTOR_MISMATCH": {
        "label"      : "하위섹터 불일치",
        "description": "실제 수혜가 ETF 내 소수 종목. ETF 전체 beta 반전 없음 → 신호 0건. COVID/XLV",
        "affects"    : ["SUBSECTOR_SPECIFIC"],
        "mitigation" : "하위 테마 ETF(XBI 등) 또는 개별 종목 직접 스크리닝으로 대체",
        "layer"      : 4,
        "color"      : "#e74c3c",
    },
    "HARKING_BIAS": {
        "label"      : "사후 가설화 편향 (HARKing)",
        "description": "결과 관찰 후 가설 설정 → 외견상 OOS지만 실질적으로는 IS 오염",
        "affects"    : ["ALL"],
        "mitigation" : "역사적 유사사건(우크라이나 2022 등) blind 사전 테스트로 검증",
        "layer"      : 4,
        "color"      : "#e74c3c",
    },
    "GARCH_CONTAMINATION": {
        "label"      : "GARCH 전기간 오염",
        "description": "VRP 계산 시 GARCH를 전기간 fit → OOS 변동성 정보 사전 반영",
        "affects"    : ["ALL"],
        "mitigation" : "EWMA(λ=0.94) 또는 expanding-window GARCH 사용",
        "layer"      : 4,
        "color"      : "#e74c3c",
    },
}

# 역사적 검증 증거
EVIDENCE = {
    "hormuz_slb"    : {"crisis": "호르무즈", "type": "SUPPLY_SHOCK",       "ticker": "SLB",      "oos_sharpe": 2.63, "supports": True},
    "hormuz_267250" : {"crisis": "호르무즈", "type": "SUPPLY_SHOCK",       "ticker": "267250.KS","oos_sharpe": 2.03, "supports": True},
    "hormuz_xom"    : {"crisis": "호르무즈", "type": "SUPPLY_SHOCK",       "ticker": "XOM",      "oos_sharpe": 1.28, "supports": True},
    "gfc_c"         : {"crisis": "GFC",     "type": "STRUCTURAL_COLLAPSE","ticker": "C",        "oos_sharpe": 1.00, "supports": True},
    "gfc_bac"       : {"crisis": "GFC",     "type": "STRUCTURAL_COLLAPSE","ticker": "BAC",      "oos_sharpe": 0.88, "supports": True},
    "dotcom_msft"   : {"crisis": "닷컴",    "type": "VALUATION_CORRECTION","ticker": "MSFT",    "oos_sharpe":-1.22, "supports": False},
    "dotcom_csco"   : {"crisis": "닷컴",    "type": "VALUATION_CORRECTION","ticker": "CSCO",    "oos_sharpe":-0.13, "supports": False},
    "covid_mrna"    : {"crisis": "COVID",   "type": "SUBSECTOR_SPECIFIC",  "ticker": "MRNA",    "oos_sharpe": np.nan,"supports": False},
}

# ===========================================================================
# 그래프 구축
# ===========================================================================

def build_ontology_graph() -> nx.DiGraph:
    G = nx.DiGraph()

    # ── Layer 1: 이벤트 유형 노드 ────────────────────────────────────────
    for eid, attrs in EVENT_TYPES.items():
        G.add_node(f"ET:{eid}", node_class="EventType", id=eid, **attrs)

    # ── Layer 2: 메커니즘 노드 ───────────────────────────────────────────
    for mid, attrs in MECHANISMS.items():
        G.add_node(f"M:{mid}", node_class="Mechanism", id=mid, **attrs)

    # ── Layer 3: 유효성 조건 노드 ────────────────────────────────────────
    for vid, attrs in VALIDITY_CONDITIONS.items():
        G.add_node(f"V:{vid}", node_class="Validity", id=vid, **attrs)

    # ── Layer 4a: 매매 결정 노드 ─────────────────────────────────────────
    for tid, attrs in TRADE_DECISIONS.items():
        G.add_node(f"TD:{tid}", node_class="TradeDecision", id=tid, **attrs)

    # ── Layer 4b: 실패 모드 노드 ─────────────────────────────────────────
    for fid, attrs in FAILURE_MODES.items():
        G.add_node(f"FM:{fid}", node_class="FailureMode", id=fid, **attrs)

    # ── 증거 노드 ────────────────────────────────────────────────────────
    for eid, attrs in EVIDENCE.items():
        G.add_node(f"EV:{eid}", node_class="Evidence", id=eid, **attrs)

    # ===========================================================================
    # 엣지 정의
    # ===========================================================================

    # ── Layer 1→2: 이벤트 유형 → 메커니즘 (인과 경로) ──────────────────
    causal_edges = [
        # 공급 충격 경로
        ("ET:SUPPLY_SHOCK",         "M:GEO_SUPPLY_CUT",         "TRIGGERS",      "지정학 이벤트가 공급 차단 유발"),
        ("M:GEO_SUPPLY_CUT",        "M:COMMODITY_PRICE_SPIKE",  "CAUSES",        "공급 감소 → 가격 급등"),
        ("M:COMMODITY_PRICE_SPIKE", "M:ETF_DECOUPLING",         "CAUSES",        "에너지주 수익 급증 → 시장과 역방향"),
        ("M:ETF_DECOUPLING",        "M:BETA_REVERSAL",          "MANIFESTS_AS",  "디커플링의 통계적 측정값"),
        ("M:BETA_REVERSAL",         "M:CAPM_IV_UNDERESTIMATE",  "CAUSES",        "beta 부호 반전 → CAPM IV 과소추정"),
        ("M:CAPM_IV_UNDERESTIMATE", "M:VRP_FLIP",               "CAUSES",        "IV < RV → VRP 음전환"),

        # 구조적 붕괴 경로
        ("ET:STRUCTURAL_COLLAPSE",  "M:SYSTEMIC_CREDIT",        "TRIGGERS",      "금융 섹터 내 연쇄 도산"),
        ("M:SYSTEMIC_CREDIT",       "M:ETF_DECOUPLING",         "CAUSES",        "금융주 beta 급등"),
        ("M:SYSTEMIC_CREDIT",       "M:VRP_FLIP",               "CAUSES",        "패닉으로 RV > IV"),

        # 실패 경로
        ("ET:VALUATION_CORRECTION", "M:BEAR_RALLY",             "PRONE_TO",      "밸류에이션 조정은 베어랠리 반복"),
        ("ET:SUBSECTOR_SPECIFIC",   "M:ETF_DECOUPLING",         "BLOCKS",        "하위섹터만 움직여 ETF 전체 반전 없음"),
    ]

    # ── Layer 2→3: 메커니즘 → 유효성 조건 ──────────────────────────────
    validity_edges = [
        ("M:ETF_DECOUPLING",  "V:FULL_ETF_DECOUPLING",    "SATISFIES",  "ETF 전체 beta 반전 = 조건 충족"),
        ("M:BETA_REVERSAL",   "V:STRUCTURAL_PERSISTENCE", "REQUIRED_BY","반전이 지속되어야 신호 유효"),
        ("M:SYSTEMIC_CREDIT", "V:STRUCTURAL_CAUSE",       "SATISFIES",  "시스템 리스크 = 구조적 원인 충족"),
        ("M:GEO_SUPPLY_CUT",  "V:STRUCTURAL_CAUSE",       "SATISFIES",  "물리적 공급 제약 = 구조적 원인 충족"),
        ("M:BEAR_RALLY",      "V:STRUCTURAL_PERSISTENCE", "VIOLATES",   "베어랠리 = 지속성 조건 위반"),
    ]

    # ── Layer 3→4: 유효성 조건 → 매매 결정 ─────────────────────────────
    trade_edges = [
        ("V:FULL_ETF_DECOUPLING",    "TD:LONG_BENEFICIARY",  "ENABLES",    "전체 디커플링 확인 시 LONG 진입 가능"),
        ("V:FULL_ETF_DECOUPLING",    "TD:SHORT_VICTIM",      "ENABLES",    "beta 급등 확인 시 SHORT 진입 가능"),
        ("V:STRUCTURAL_PERSISTENCE", "TD:LONG_BENEFICIARY",  "REQUIRED_BY","지속성 없으면 LONG 진입 불가"),
        ("V:STRUCTURAL_PERSISTENCE", "TD:SHORT_VICTIM",      "REQUIRED_BY","지속성 없으면 SHORT 진입 불가"),
        ("V:STRUCTURAL_CAUSE",       "TD:LONG_BENEFICIARY",  "REQUIRED_BY","구조적 원인 없으면 LONG 신뢰 불가"),
        ("V:STRUCTURAL_CAUSE",       "TD:SHORT_VICTIM",      "REQUIRED_BY","구조적 원인 없으면 SHORT 신뢰 불가"),
    ]

    # ── 이벤트 유형 → 매매 결정 (직접 판단) ────────────────────────────
    decision_edges = [
        ("ET:SUPPLY_SHOCK",         "TD:LONG_BENEFICIARY",  "RECOMMENDS", "공급 충격 수혜 섹터 매수"),
        ("ET:STRUCTURAL_COLLAPSE",  "TD:SHORT_VICTIM",      "RECOMMENDS", "붕괴 섹터 공매도"),
        ("ET:VALUATION_CORRECTION", "TD:AVOID",             "RECOMMENDS", "신호 타이밍 실패 위험 → 적용 불가"),
        ("ET:SUBSECTOR_SPECIFIC",   "TD:MONITOR",           "RECOMMENDS", "하위ETF 또는 개별종목 직접 탐색으로 전환"),
    ]

    # ── 실패 모드 연결 ───────────────────────────────────────────────────
    failure_edges = [
        ("ET:VALUATION_CORRECTION", "FM:BEAR_RALLY_SQUEEZE",  "PRONE_TO",     "밸류에이션 조정에서 랠리 스퀴즈 위험"),
        ("ET:SUBSECTOR_SPECIFIC",   "FM:SUBSECTOR_MISMATCH",  "PRONE_TO",     "하위섹터 불일치 위험"),
        ("TD:LONG_BENEFICIARY",     "FM:HARKING_BIAS",        "VULNERABLE_TO","모든 LONG 결정에 사후편향 위험"),
        ("TD:SHORT_VICTIM",         "FM:HARKING_BIAS",        "VULNERABLE_TO","모든 SHORT 결정에 사후편향 위험"),
        ("TD:LONG_BENEFICIARY",     "FM:GARCH_CONTAMINATION", "VULNERABLE_TO","VRP 계산에 GARCH 오염 위험"),
        ("TD:SHORT_VICTIM",         "FM:GARCH_CONTAMINATION", "VULNERABLE_TO","VRP 계산에 GARCH 오염 위험"),
        ("FM:BEAR_RALLY_SQUEEZE",   "TD:SHORT_VICTIM",        "DEGRADES",     "랠리 스퀴즈로 SHORT 성과 저하"),
        ("FM:SUBSECTOR_MISMATCH",   "TD:LONG_BENEFICIARY",    "DEGRADES",     "불일치로 신호 불발"),
    ]

    # ── 증거 → 이벤트 유형 (검증/반증) ─────────────────────────────────
    evidence_edges = []
    type_map = {
        "SUPPLY_SHOCK":       "ET:SUPPLY_SHOCK",
        "STRUCTURAL_COLLAPSE":"ET:STRUCTURAL_COLLAPSE",
        "VALUATION_CORRECTION":"ET:VALUATION_CORRECTION",
        "SUBSECTOR_SPECIFIC": "ET:SUBSECTOR_SPECIFIC",
    }
    for eid, ev in EVIDENCE.items():
        rel = "CONFIRMS" if ev["supports"] else "REFUTES"
        evidence_edges.append((
            f"EV:{eid}",
            type_map[ev["type"]],
            rel,
            f"{ev['ticker']} OOS Sharpe={ev['oos_sharpe']:.2f}" if not np.isnan(ev.get("oos_sharpe", np.nan)) else f"{ev['ticker']} (신호 불발)"
        ))

    # 모든 엣지 추가
    for src, dst, rel, reason in (causal_edges + validity_edges + trade_edges +
                                   decision_edges + failure_edges + evidence_edges):
        G.add_edge(src, dst, relation=rel, reason=reason)

    return G


# ===========================================================================
# 추론 엔진
# ===========================================================================

@dataclass
class ClassificationResult:
    event_name   : str
    event_type   : str
    applicable   : bool
    direction    : str
    action       : str
    signals      : list = field(default_factory=list)
    validity_ok  : list = field(default_factory=list)
    validity_fail: list = field(default_factory=list)
    failure_risks: list = field(default_factory=list)
    reasoning    : list = field(default_factory=list)
    confidence   : str  = "LOW"
    similar_ev   : list = field(default_factory=list)


def classify_event(
    G             : nx.DiGraph,
    event_name    : str,
    commodity_driven    : bool,
    external_physical   : bool,
    sector_is_epicenter : bool,
    valuation_driven    : bool,
    subsector_only      : bool,
    full_etf_decoupling : bool,
    structural_cause    : bool,
    expected_duration_d : int,
) -> ClassificationResult:
    """
    새 이벤트의 특성을 입력받아 온톨로지 추론으로 매매 결정 도출.

    Parameters
    ----------
    commodity_driven     : 상품(원유/가스/반도체 등) 가격이 핵심 드라이버인가
    external_physical    : 지정학/물리적 외부 원인인가 (내부 밸류에이션 아님)
    sector_is_epicenter  : 해당 섹터가 위기의 진원지인가 (금융위기의 은행)
    valuation_driven     : 밸류에이션 조정이 주 원인인가
    subsector_only       : ETF 내 일부 종목만 영향받는가
    full_etf_decoupling  : ETF 전체 beta 반전이 관찰/예상되는가
    structural_cause     : 물리적/제도적 구조 원인인가
    expected_duration_d  : 체제 전환 지속 예상 기간 (거래일)
    """
    r = ClassificationResult(event_name=event_name, event_type="UNKNOWN",
                             applicable=False, direction="NONE", action="NEUTRAL")

    # ── Step 1: 이벤트 유형 분류 ─────────────────────────────────────────
    if subsector_only:
        r.event_type = "SUBSECTOR_SPECIFIC"
        r.reasoning.append("ETF 내 소수 종목만 영향 → 하위섹터 특수 분류")
    elif valuation_driven and not external_physical:
        r.event_type = "VALUATION_CORRECTION"
        r.reasoning.append("밸류에이션 드라이버 + 외부 충격 없음 → 밸류에이션 조정 분류")
    elif commodity_driven and external_physical and not sector_is_epicenter:
        r.event_type = "SUPPLY_SHOCK"
        r.reasoning.append("상품 드라이버 + 외부 물리적 원인 → 공급 충격 분류")
    elif sector_is_epicenter and not commodity_driven:
        r.event_type = "STRUCTURAL_COLLAPSE"
        r.reasoning.append("섹터가 위기 진원지 → 구조적 붕괴 분류")
    else:
        r.event_type = "SUPPLY_SHOCK"  # default for geopolitical
        r.reasoning.append("분류 기준 일부 미충족 — 공급 충격으로 잠정 분류 (재검토 권장)")

    et = EVENT_TYPES[r.event_type]
    r.applicable = et["framework_applicable"]
    r.direction  = et["direction"]

    if not r.applicable:
        r.action = TRADE_DECISIONS["AVOID"]["action"] if r.event_type == "VALUATION_CORRECTION" \
                   else TRADE_DECISIONS["MONITOR"]["action"]
        fail = "BEAR_RALLY_SQUEEZE" if r.event_type == "VALUATION_CORRECTION" \
               else "SUBSECTOR_MISMATCH"
        r.failure_risks.append(FAILURE_MODES[fail]["label"])
        r.reasoning.append(f"프레임워크 적용 불가: {et['description']}")
        r.confidence = "HIGH"  # 실패임을 높은 신뢰도로 판단
        return r

    # ── Step 2: 유효성 조건 확인 ─────────────────────────────────────────
    checks = {
        "FULL_ETF_DECOUPLING" : full_etf_decoupling,
        "STRUCTURAL_PERSISTENCE": expected_duration_d >= 20,
        "STRUCTURAL_CAUSE"    : structural_cause,
    }
    for cid, passed in checks.items():
        vc = VALIDITY_CONDITIONS[cid]
        if passed:
            r.validity_ok.append(vc["label"])
            r.reasoning.append(f"[OK] {vc['label']}: 충족")
        else:
            r.validity_fail.append(vc["label"])
            r.reasoning.append(f"[FAIL] {vc['label']}: 미충족 ({vc['how_to_check']})")

    # ── Step 3: 최종 매매 결정 ───────────────────────────────────────────
    n_fail = len(r.validity_fail)
    if n_fail == 0:
        r.action    = r.direction  # LONG or SHORT
        r.confidence = "HIGH"
        r.signals = (["S2_VRP (VRP < 0 연속 3일)", "S3_COMBO (beta + VRP 동시)"]
                     if r.direction == "LONG"
                     else ["S3_COMBO (beta > 1.5 AND VRP < 0 연속 3일)"])
        r.reasoning.append(f"유효성 조건 모두 충족 → {r.direction} 신호 활성화")
    elif n_fail == 1:
        r.action    = "MONITOR"
        r.confidence = "MEDIUM"
        r.signals   = ["S1_BETA 관찰 (포지션 없이 모니터링)"]
        r.reasoning.append(f"유효성 조건 {n_fail}개 미충족 → 포지션 없이 MONITOR")
    else:
        r.action    = "NEUTRAL"
        r.confidence = "LOW"
        r.reasoning.append(f"유효성 조건 {n_fail}개 미충족 → 신호 신뢰 불가")

    # ── Step 4: 실패 모드 경고 ───────────────────────────────────────────
    r.failure_risks.append(FAILURE_MODES["HARKING_BIAS"]["label"])
    r.failure_risks.append(FAILURE_MODES["GARCH_CONTAMINATION"]["label"])

    # ── Step 5: 유사 사례 검색 ───────────────────────────────────────────
    for eid, ev in EVIDENCE.items():
        if ev["type"] == r.event_type and ev["supports"]:
            sharpe_str = (f"Sharpe={ev['oos_sharpe']:.2f}"
                          if not np.isnan(ev.get("oos_sharpe", np.nan))
                          else "신호불발")
            r.similar_ev.append(f"{ev['crisis']}/{ev['ticker']} ({sharpe_str})")

    return r


def print_classification(res: ClassificationResult):
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  이벤트: {res.event_name}")
    print(f"  분류  : {EVENT_TYPES[res.event_type]['label']} ({res.event_type})")
    print(f"  적용  : {'가능' if res.applicable else '불가'}")
    print(f"  방향  : {res.direction}  |  결정: {res.action}  |  신뢰도: {res.confidence}")
    print(sep)
    print("  [추론 과정]")
    for step in res.reasoning:
        print(f"    {step}")
    if res.signals:
        print("\n  [모니터링할 신호]")
        for s in res.signals:
            print(f"    - {s}")
    if res.validity_fail:
        print("\n  [미충족 유효성 조건]")
        for v in res.validity_fail:
            print(f"    - {v}")
    if res.failure_risks:
        print("\n  [실패 모드 경고]")
        for f in res.failure_risks:
            print(f"    - {f}")
    if res.similar_ev:
        print("\n  [유사 검증 사례]")
        for e in res.similar_ev:
            print(f"    - {e}")
    print(sep)


# ===========================================================================
# 시각화
# ===========================================================================

LAYER_X = {1: 0.10, 2: 0.34, 3: 0.60, 4: 0.86}
LAYER_LABELS = {
    1: "Layer 1\n이벤트 유형",
    2: "Layer 2\n인과 메커니즘",
    3: "Layer 3\n유효성 조건",
    4: "Layer 4\n매매 결정",
}

NODE_Y_POSITIONS = {
    # Layer 1
    "ET:SUPPLY_SHOCK"         : (LAYER_X[1], 0.82),
    "ET:STRUCTURAL_COLLAPSE"  : (LAYER_X[1], 0.58),
    "ET:VALUATION_CORRECTION" : (LAYER_X[1], 0.35),
    "ET:SUBSECTOR_SPECIFIC"   : (LAYER_X[1], 0.12),

    # Layer 2 (mechanisms)
    "M:GEO_SUPPLY_CUT"        : (LAYER_X[2], 0.92),
    "M:COMMODITY_PRICE_SPIKE" : (LAYER_X[2], 0.80),
    "M:ETF_DECOUPLING"        : (LAYER_X[2], 0.68),
    "M:BETA_REVERSAL"         : (LAYER_X[2], 0.56),
    "M:CAPM_IV_UNDERESTIMATE" : (LAYER_X[2], 0.44),
    "M:VRP_FLIP"              : (LAYER_X[2], 0.33),
    "M:SYSTEMIC_CREDIT"       : (LAYER_X[2], 0.21),
    "M:BEAR_RALLY"            : (LAYER_X[2], 0.09),

    # Layer 3
    "V:FULL_ETF_DECOUPLING"   : (LAYER_X[3], 0.80),
    "V:STRUCTURAL_PERSISTENCE": (LAYER_X[3], 0.55),
    "V:STRUCTURAL_CAUSE"      : (LAYER_X[3], 0.30),

    # Layer 4 — trade decisions
    "TD:LONG_BENEFICIARY"     : (LAYER_X[4], 0.82),
    "TD:SHORT_VICTIM"         : (LAYER_X[4], 0.62),
    "TD:MONITOR"              : (LAYER_X[4], 0.42),
    "TD:AVOID"                : (LAYER_X[4], 0.22),

    # Layer 4 — failure modes (slightly right)
    "FM:BEAR_RALLY_SQUEEZE"   : (LAYER_X[4] + 0.10, 0.12),
    "FM:SUBSECTOR_MISMATCH"   : (LAYER_X[4] + 0.10, 0.04),
}

NODE_CLASS_COLOR = {
    "EventType"   : "#2ecc71",
    "Mechanism"   : "#3498db",
    "Validity"    : "#9b59b6",
    "TradeDecision": "#e67e22",
    "FailureMode" : "#e74c3c",
    "Evidence"    : "#95a5a6",
}
NODE_CLASS_COLOR_OVERRIDES = {
    "ET:SUPPLY_SHOCK"         : "#27ae60",
    "ET:STRUCTURAL_COLLAPSE"  : "#c0392b",
    "ET:VALUATION_CORRECTION" : "#e67e22",
    "ET:SUBSECTOR_SPECIFIC"   : "#7f8c8d",
    "TD:LONG_BENEFICIARY"     : "#27ae60",
    "TD:SHORT_VICTIM"         : "#c0392b",
    "TD:MONITOR"              : "#f39c12",
    "TD:AVOID"                : "#7f8c8d",
}

EDGE_RELATION_STYLE = {
    "TRIGGERS"      : {"color": "#2ecc71", "style": "-",   "width": 1.8},
    "CAUSES"        : {"color": "#3498db", "style": "-",   "width": 1.5},
    "MANIFESTS_AS"  : {"color": "#3498db", "style": "--",  "width": 1.2},
    "SATISFIES"     : {"color": "#9b59b6", "style": "-",   "width": 1.5},
    "REQUIRED_BY"   : {"color": "#9b59b6", "style": "--",  "width": 1.2},
    "RECOMMENDS"    : {"color": "#e67e22", "style": "-",   "width": 2.0},
    "ENABLES"       : {"color": "#27ae60", "style": "--",  "width": 1.2},
    "PRONE_TO"      : {"color": "#e74c3c", "style": ":",   "width": 1.2},
    "DEGRADES"      : {"color": "#e74c3c", "style": "-",   "width": 1.2},
    "VIOLATES"      : {"color": "#e74c3c", "style": ":",   "width": 1.5},
    "BLOCKS"        : {"color": "#e74c3c", "style": "-",   "width": 1.5},
    "VULNERABLE_TO" : {"color": "#e74c3c", "style": ":",   "width": 1.0},
    "CONFIRMS"      : {"color": "#27ae60", "style": "--",  "width": 1.0},
    "REFUTES"       : {"color": "#e74c3c", "style": ":",   "width": 1.0},
}


def plot_ontology(G: nx.DiGraph):
    # 위치 계산: 고정 좌표 + 나머지 자동
    pos = {}
    for nid in G.nodes():
        if nid in NODE_Y_POSITIONS:
            pos[nid] = NODE_Y_POSITIONS[nid]

    # Evidence 노드 위치 (하단 분산)
    ev_nodes = [n for n in G.nodes() if n.startswith("EV:")]
    for i, n in enumerate(ev_nodes):
        pos[n] = (0.02 + i * 0.12, 0.01)

    fig, ax = plt.subplots(figsize=(22, 13))
    ax.set_xlim(-0.02, 1.12)
    ax.set_ylim(-0.05, 1.08)
    ax.axis("off")

    # 레이어 배경
    layer_colors = ["#f0faf0", "#eaf4fb", "#f5eefb", "#fef9ec"]
    for li, (lx, lname) in enumerate(LAYER_LABELS.items()):
        cx = LAYER_X[li + 1]
        rect = plt.Rectangle((cx - 0.10, -0.02), 0.20, 1.04,
                              linewidth=1.2, edgecolor="#cccccc",
                              facecolor=layer_colors[li], alpha=0.4, zorder=0)
        ax.add_patch(rect)
        ax.text(cx, 1.05, lname, ha="center", va="bottom",
                fontsize=9, weight="bold", color="#444444")

    # 엣지 그리기 (evidence 제외)
    drawn_relations = set()
    for src, dst, data in G.edges(data=True):
        if src.startswith("EV:") or dst.startswith("EV:"):
            continue
        if src not in pos or dst not in pos:
            continue
        rel   = data.get("relation", "CAUSES")
        style = EDGE_RELATION_STYLE.get(rel, {"color": "#aaaaaa", "style": "-", "width": 1.0})
        x0, y0 = pos[src]
        x1, y1 = pos[dst]

        arrowprops = dict(
            arrowstyle="-|>",
            color=style["color"],
            lw=style["width"],
            linestyle=style["style"],
            connectionstyle="arc3,rad=0.05",
        )
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=arrowprops, zorder=1)
        drawn_relations.add(rel)

    # 노드 그리기
    node_shapes = {
        "EventType"    : "s",   # square
        "Mechanism"    : "o",   # circle
        "Validity"     : "D",   # diamond
        "TradeDecision": "^",   # triangle
        "FailureMode"  : "X",   # X
        "Evidence"     : ".",   # small dot
    }
    node_sizes  = {"EventType": 900, "Mechanism": 700, "Validity": 700,
                   "TradeDecision": 900, "FailureMode": 700, "Evidence": 300}

    for nid, ndata in G.nodes(data=True):
        if nid not in pos:
            continue
        nc   = ndata.get("node_class", "Mechanism")
        col  = NODE_CLASS_COLOR_OVERRIDES.get(nid,
               NODE_CLASS_COLOR.get(nc, "#aaaaaa"))
        x, y = pos[nid]
        size = node_sizes.get(nc, 600)
        ax.scatter(x, y, s=size, c=col, zorder=3,
                   marker=node_shapes.get(nc, "o"),
                   edgecolors="white", linewidths=1.5)

        # 노드 레이블
        label = ndata.get("label", nid.split(":")[-1])
        if len(label) > 14:
            label = label[:12] + "…"
        fontsize = 7.5 if nc == "Evidence" else 8.5
        ax.text(x, y - 0.045, label, ha="center", va="top",
                fontsize=fontsize, color="#222222",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                          edgecolor="none", alpha=0.7))

    # 범례
    legend_entries = [
        mpatches.Patch(color="#27ae60", label="공급 충격 / LONG 수혜"),
        mpatches.Patch(color="#c0392b", label="구조적 붕괴 / SHORT 피해"),
        mpatches.Patch(color="#e67e22", label="밸류에이션 조정 (불가)"),
        mpatches.Patch(color="#7f8c8d", label="하위섹터 특수 (불가)"),
        mpatches.Patch(color="#3498db", label="메커니즘"),
        mpatches.Patch(color="#9b59b6", label="유효성 조건"),
        mpatches.Patch(color="#e74c3c", label="실패 모드"),
    ]
    ax.legend(handles=legend_entries, loc="lower right",
              fontsize=8, framealpha=0.9)

    # 유효성 조건 요약 텍스트
    summary_text = (
        "신호 유효 조건 (3가지 모두 필요):\n"
        "  1. 섹터 ETF 전체 디커플링 (일부 종목 아님)\n"
        "  2. 구조적 지속성 (> 20 거래일)\n"
        "  3. 구조적(물리적/제도적) 원인"
    )
    ax.text(0.01, 0.99, summary_text, transform=ax.transAxes,
            va="top", ha="left", fontsize=8.5,
            bbox=dict(boxstyle="round", facecolor="#fffde7",
                      edgecolor="#f39c12", alpha=0.9))

    # 검증 증거 요약
    ev_summary = "검증 증거:\n"
    for eid, ev in EVIDENCE.items():
        mark   = "✓" if ev["supports"] else "✗"
        sharpe = f"{ev['oos_sharpe']:.2f}" if not np.isnan(ev.get("oos_sharpe", np.nan)) else "n/a"
        ev_summary += f"  {mark} {ev['crisis']}/{ev['ticker']} Sharpe={sharpe}\n"
    ax.text(0.01, 0.62, ev_summary, transform=ax.transAxes,
            va="top", ha="left", fontsize=7.5,
            bbox=dict(boxstyle="round", facecolor="#f8f8f8",
                      edgecolor="#cccccc", alpha=0.9))

    ax.set_title(
        "트레이딩 온톨로지: Beta 체제전환 + VRP 신호 프레임워크\n"
        "4개 역사적 위기(닷컴/GFC/COVID/호르무즈) 검증 결과 기반",
        fontsize=13, weight="bold", pad=12,
    )

    out = FIGURES_DIR / "trading_ontology.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  차트 저장: {out}")


# ===========================================================================
# JSON-LD 내보내기
# ===========================================================================

def export_jsonld(G: nx.DiGraph):
    context = {
        "@context": {
            "schema"   : "https://schema.org/",
            "trading"  : "https://trading-ontology.local/",
            "label"    : "schema:name",
            "layer"    : "trading:layer",
            "relation" : "trading:relation",
        }
    }

    nodes_out = []
    for nid, ndata in G.nodes(data=True):
        node_obj = {"@id": nid, "@type": ndata.get("node_class", "Node")}
        for k, v in ndata.items():
            if k not in ("color",) and isinstance(v, (str, int, float, bool, list)):
                node_obj[k] = v
        nodes_out.append(node_obj)

    edges_out = []
    for src, dst, edata in G.edges(data=True):
        edges_out.append({
            "@type"   : "Edge",
            "source"  : src,
            "target"  : dst,
            "relation": edata.get("relation", ""),
            "reason"  : edata.get("reason", ""),
        })

    payload = {**context, "nodes": nodes_out, "edges": edges_out}
    out = OUTPUT_DIR / "ontology_trading.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"  JSON 저장: {out}")


# ===========================================================================
# 메인
# ===========================================================================

if __name__ == "__main__":
    sep = "=" * 62
    print(f"\n{sep}")
    print("  트레이딩 온톨로지 구축 + 추론 엔진")
    print(f"{sep}")

    # 1) 그래프 구축
    print("\n[1] 온톨로지 그래프 구축...")
    G = build_ontology_graph()
    print(f"  노드 수: {G.number_of_nodes()}  |  엣지 수: {G.number_of_edges()}")

    # 2) JSON-LD 내보내기
    print("\n[2] JSON-LD 내보내기...")
    export_jsonld(G)

    # 3) 시각화
    print("\n[3] 시각화...")
    plot_ontology(G)

    # 4) 추론: 역사적 사례 재검증 (온톨로지가 올바르게 분류하는지 확인)
    print(f"\n[4] 역사적 사례 분류 재현 (검증)")

    test_cases = [
        dict(
            event_name="호르무즈 봉쇄 (2025)",
            commodity_driven=True,
            external_physical=True,
            sector_is_epicenter=False,
            valuation_driven=False,
            subsector_only=False,
            full_etf_decoupling=True,
            structural_cause=True,
            expected_duration_d=120,
        ),
        dict(
            event_name="글로벌 금융위기 (2008)",
            commodity_driven=False,
            external_physical=False,
            sector_is_epicenter=True,
            valuation_driven=False,
            subsector_only=False,
            full_etf_decoupling=True,
            structural_cause=True,
            expected_duration_d=180,
        ),
        dict(
            event_name="닷컴버블 붕괴 (2000)",
            commodity_driven=False,
            external_physical=False,
            sector_is_epicenter=False,
            valuation_driven=True,
            subsector_only=False,
            full_etf_decoupling=False,
            structural_cause=False,
            expected_duration_d=700,
        ),
        dict(
            event_name="COVID-19 팬데믹 (2020)",
            commodity_driven=False,
            external_physical=True,
            sector_is_epicenter=False,
            valuation_driven=False,
            subsector_only=True,
            full_etf_decoupling=False,
            structural_cause=True,
            expected_duration_d=300,
        ),
    ]

    for tc in test_cases:
        res = classify_event(G, **tc)
        print_classification(res)

    # 5) 새 이벤트 추론 예시: 대만 해협 위기
    print(f"\n[5] 신규 이벤트 추론: 대만 해협 반도체 공급 위기 (가상)")
    new_event = classify_event(
        G,
        event_name="대만 해협 봉쇄 (가상 시나리오)",
        commodity_driven=True,        # 반도체 = 현대판 전략 상품
        external_physical=True,       # 군사적 봉쇄
        sector_is_epicenter=False,    # 섹터가 위기 진원지 아님
        valuation_driven=False,
        subsector_only=False,         # XLK 전체가 영향 (TSMC 의존도 높음)
        full_etf_decoupling=True,     # 반도체 쇼크로 XLK 전체 beta 반전 예상
        structural_cause=True,        # 물리적 공급 차단
        expected_duration_d=90,
    )
    print_classification(new_event)

    print(f"\n{sep}")
    print("  완료. 출력 파일:")
    print(f"    output/ontology_trading.json")
    print(f"    output/figures/trading_ontology.png")
    print(f"{sep}\n")

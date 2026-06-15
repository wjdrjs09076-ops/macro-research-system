"""
build_overview_pdf.py — 매크로 리서치 시스템 핵심 아이디어 흐름도 PDF 생성.

페이지 구성 (15p, 2026-06-12)
─────────
1. 표지 — 비전 + 핵심 질문
2. 전체 흐름도 — 6-Layer + Kinetic Layer
3. Layer 1: GARCH(1,1) — 변동성과 VRP
4. Layer 2: EVT/GPD — Fat-tail
5. Layer 3: Copula — 꼬리 의존성
6. Layer 4: Cholesky 분산 분해 — 실제 XLK 분해 그래프
7. PCMCI 인과 체인
8. Layer 5: Ontology Gate — 8 룰
9. 민감도 감사 — raw vs ctrl delta (NEW 06-12, 라이브 데이터)
10. NORTA Joint Drawdown — 시장 base rate (실제 시뮬)
11. 트리플 전략 — 같은 시그널 세 베팅
12. Kinetic Layer 피드백 루프 + 수명주기 가드
13. 백테스트 결과
14. 알려진 한계
15. 용어집

실제 데이터 사용: sector_returns.parquet, ontology_signals.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib import rcParams

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUTPUT_DIR

# ── 한글 폰트 ─────────────────────────────────────────────────
rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
# monospace 호출 시 한글 깨짐 방지. Malgun Gothic 을 첫째로 두어 mixed text 도 안전.
# 코드/수식 미관은 약간 손해지만 PDF 가독성 우선.
rcParams["font.monospace"] = ["Malgun Gothic", "Consolas", "DejaVu Sans Mono"]
rcParams["axes.unicode_minus"] = False
rcParams["pdf.fonttype"] = 42   # TrueType embedding (텍스트 추출 가능)

PDF_OUT = Path(__file__).resolve().parent.parent / "macro_research_overview.pdf"

# ── 색상 팔레트 (잉크 절약 + 가독성) ───────────────────────────
C_BG       = "#0b1220"
C_PANEL    = "#1f2937"
C_TEXT     = "#e5e7eb"
C_MUTED    = "#9ca3af"
C_ACCENT_1 = "#8b5cf6"  # violet (Layer/스트래들)
C_ACCENT_2 = "#06b6d4"  # cyan (방향성)
C_ACCENT_3 = "#10b981"  # emerald (Kinetic/OK)
C_ACCENT_4 = "#f59e0b"  # amber (변동성/주의)
C_ACCENT_5 = "#ef4444"  # red (꼬리/위험)
C_LIGHT    = "#374151"


# ──────────────────────────────────────────────────────────────
# 공용 헬퍼
# ──────────────────────────────────────────────────────────────

PAGE_SIZE = (12.0, 9.0)


def _new_page(figsize=None):
    """모든 페이지를 동일한 크기(12x9 인치)로 생성. PDF 일관성을 위해 통일."""
    if figsize is None:
        figsize = PAGE_SIZE
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(C_BG)
    return fig


def _title(fig, title, subtitle=None):
    fig.text(0.04, 0.93, title, color=C_TEXT, fontsize=22, fontweight="bold")
    if subtitle:
        fig.text(0.04, 0.89, subtitle, color=C_MUTED, fontsize=11)


def _footer(fig, n, total):
    fig.text(0.96, 0.03, f"{n} / {total}", color=C_MUTED, fontsize=9, ha="right")
    fig.text(0.04, 0.03, "Macro Research System  ·  2026-06-16",
             color=C_MUTED, fontsize=9)


def _box(ax, x, y, w, h, text, *, fc=C_PANEL, ec=C_LIGHT, fontcolor=C_TEXT,
         fontsize=10, bold=False, badge=None, badge_fc=C_ACCENT_1):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                       linewidth=1.2, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    weight = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=fontcolor, fontsize=fontsize, fontweight=weight,
            wrap=True, zorder=3)
    if badge:
        bw = 0.08
        bp = FancyBboxPatch((x + w - bw - 0.005, y + h - 0.07),
                            bw, 0.06,
                            boxstyle="round,pad=0.005,rounding_size=0.02",
                            linewidth=0, facecolor=badge_fc, zorder=4)
        ax.add_patch(bp)
        ax.text(x + w - bw / 2 - 0.005, y + h - 0.04, badge,
                ha="center", va="center", color="white", fontsize=8,
                fontweight="bold", zorder=5)


def _arrow(ax, x1, y1, x2, y2, *, color=C_MUTED, label=None, lw=1.4):
    ar = FancyArrowPatch((x1, y1), (x2, y2),
                         arrowstyle="-|>", mutation_scale=14,
                         color=color, lw=lw, zorder=1)
    ax.add_patch(ar)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.012, label,
                ha="center", va="bottom", color=C_MUTED, fontsize=8)


def _blank_axes(fig, rect=(0.04, 0.06, 0.92, 0.80)):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_facecolor(C_BG)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    return ax


# ──────────────────────────────────────────────────────────────
# Page 1 — Cover
# ──────────────────────────────────────────────────────────────

def page_cover(pdf, total):
    fig = _new_page()
    ax = _blank_axes(fig, (0, 0, 1, 1))

    ax.text(0.5, 0.78, "Macro Research System",
            ha="center", color=C_TEXT, fontsize=34, fontweight="bold")
    ax.text(0.5, 0.72, "거시 변동성 감지 → 트리플 전략 자동 실행 → 사후 사이징 보정",
            ha="center", color=C_ACCENT_2, fontsize=14)

    # Central question
    ax.text(0.5, 0.58,
            "\"방향을 맞추지 않고도, 이벤트가 발생했을 때\n확률적으로 수익이 나는 구조를 만들 수 있는가?\"",
            ha="center", color=C_TEXT, fontsize=16, style="italic",
            linespacing=1.6)

    # Three pillars
    box_y = 0.30
    box_w = 0.24
    box_h = 0.14
    gap = 0.04
    total_w = box_w * 3 + gap * 2
    x0 = (1 - total_w) / 2
    pillars = [
        ("① 감지", "6-Layer 온톨로지\n(GARCH→EVT→Copula→Cholesky→Gate · ML=shadow)",
         C_ACCENT_1),
        ("② 실행", "키네틱 레이어\n→ β헤지 스프레드(R8) · event_vol 롱 볼(R9)",
         C_ACCENT_2),
        ("③ 학습", "거래 저널 → (룰×전략) 귀인\n→ 사이징 피드백",
         C_ACCENT_3),
    ]
    for i, (head, body, color) in enumerate(pillars):
        x = x0 + i * (box_w + gap)
        p = FancyBboxPatch((x, box_y), box_w, box_h,
                           boxstyle="round,pad=0.02,rounding_size=0.04",
                           linewidth=1.6, edgecolor=color, facecolor=C_PANEL)
        ax.add_patch(p)
        ax.text(x + box_w / 2, box_y + box_h - 0.035, head,
                ha="center", va="center", color=color, fontsize=14,
                fontweight="bold")
        ax.text(x + box_w / 2, box_y + box_h / 2 - 0.02, body,
                ha="center", va="center", color=C_TEXT, fontsize=10,
                linespacing=1.5)

    ax.text(0.5, 0.18,
            "방향 예측 대신 변동성을 베팅 + 같은 시그널을 크기·방향 둘로 직교 채점",
            ha="center", color=C_MUTED, fontsize=11)

    ax.text(0.5, 0.10, "2026-06-16 · CONFIG FREEZE v1 (R9 event_vol 반영) — 룰·게이트·청산·캡 동결. "
            "OOS 전향 시계 = 이 날짜부터, 페이퍼 표본만이 진짜 검증",
            ha="center", color=C_ACCENT_3, fontsize=10, fontweight="bold")

    _footer(fig, 1, total)
    pdf.savefig(fig, facecolor=C_BG); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Page 2 — Full system flow
# ──────────────────────────────────────────────────────────────

def page_flow(pdf, n, total):
    fig = _new_page(figsize=(12.0, 9.0))
    _title(fig, "전체 흐름도",
           "데이터 → 6개 분석 레이어 → 듀얼 전략 액션 → 거래 저널 → 사이징 피드백 (루프 폐쇄)")
    ax = _blank_axes(fig, (0.03, 0.06, 0.94, 0.80))

    # Data sources (top row)
    sources = [("yfinance", "섹터 ETF / VIX / 매크로"),
               ("FRED", "EPU/HY/CPI/FedFunds"),
               ("EIA", "WTI/Brent 현물·재고"),
               ("Alpaca", "페이퍼 계정·체결")]
    sx0, sy = 0.04, 0.91
    sw, sh = 0.21, 0.07
    for i, (name, sub) in enumerate(sources):
        x = sx0 + i * (sw + 0.014)
        p = FancyBboxPatch((x, sy), sw, sh,
                           boxstyle="round,pad=0.01,rounding_size=0.03",
                           linewidth=1.0, edgecolor=C_LIGHT, facecolor="#111827")
        ax.add_patch(p)
        ax.text(x + sw / 2, sy + sh - 0.022, name, ha="center", va="center",
                color=C_ACCENT_4, fontsize=10, fontweight="bold")
        ax.text(x + sw / 2, sy + 0.018, sub, ha="center", va="center",
                color=C_MUTED, fontsize=8)

    # 6 Layers (vertical stack on left)
    layers = [
        ("Layer 1", "GARCH(1,1)",                 "조건부 σ_t · VRP",          C_ACCENT_4),
        ("Layer 2", "EVT / GPD",                  "꼬리 ξ · VaR/ES(99%)",      C_ACCENT_5),
        ("Layer 3", "Copula 꼬리 의존성",         "λ_L · λ_U",                  C_ACCENT_5),
        ("Layer 4", "Multivariate MC + Cholesky", "var_attr · CO_MOVES_WITH",  C_ACCENT_2),
        ("Layer 5", "Ontology Gate",              "9 룰 → SignalNode",         C_ACCENT_1),
        ("Layer 6", "ML (XGBoost + VQC) — 미통합", "룰 입력 없음 (검증 전)",    "#6b7280"),
    ]
    lx, ly0 = 0.05, 0.78
    lw, lh = 0.40, 0.07
    gap = 0.012
    for i, (tag, name, sub, color) in enumerate(layers):
        y = ly0 - i * (lh + gap)
        p = FancyBboxPatch((lx, y), lw, lh,
                           boxstyle="round,pad=0.01,rounding_size=0.025",
                           linewidth=1.4, edgecolor=color, facecolor=C_PANEL)
        ax.add_patch(p)
        ax.text(lx + 0.018, y + lh / 2 + 0.012, tag, ha="left", va="center",
                color=color, fontsize=9, fontweight="bold")
        ax.text(lx + 0.018, y + lh / 2 - 0.013, name, ha="left", va="center",
                color=C_TEXT, fontsize=11, fontweight="bold")
        ax.text(lx + lw - 0.018, y + lh / 2, sub, ha="right", va="center",
                color=C_MUTED, fontsize=8.5)

        # 인접 레이어 연결 화살표
        if i > 0:
            _arrow(ax, lx + lw / 2, y + lh + gap, lx + lw / 2, y + lh,
                   color=C_LIGHT, lw=0.8)

    # 데이터 → Layer 1 연결
    _arrow(ax, 0.5, sy, lx + lw / 2, ly0 + lh, color=C_LIGHT, lw=0.8)

    # Kinetic Layer (right side)
    kx, ky0 = 0.55, 0.78
    kw, kh = 0.40, 0.07
    kinetic = [
        ("trigger.py",          "Signal → 듀얼 후보 매핑",         C_ACCENT_1),
        ("kinetic_executor.py", "Alpaca 진입/청산 (RTH 가드)",     C_ACCENT_3),
        ("trade_journal.py",    "JSONL · trade_id 매칭",           "#fbbf24"),
        ("attribution.py",      "(rule × strategy) 귀인",          C_ACCENT_2),
        ("feedback.py",         "→ rule_performance.json",         C_ACCENT_3),
    ]
    for i, (name, sub, color) in enumerate(kinetic):
        y = ky0 - i * (kh + gap) - 0.02
        p = FancyBboxPatch((kx, y), kw, kh,
                           boxstyle="round,pad=0.01,rounding_size=0.025",
                           linewidth=1.4, edgecolor=color, facecolor=C_PANEL)
        ax.add_patch(p)
        ax.text(kx + 0.018, y + lh / 2 + 0.012, name, ha="left", va="center",
                color=color, fontsize=10, fontweight="bold")
        ax.text(kx + 0.018, y + lh / 2 - 0.013, sub, ha="left", va="center",
                color=C_MUTED, fontsize=8.5)
        if i > 0:
            _arrow(ax, kx + kw / 2, y + lh + gap, kx + kw / 2, y + lh,
                   color=C_LIGHT, lw=0.8)

    # Layer 5 → Kinetic trigger 연결 (라벨은 두 박스 사이 빈공간 중앙에 별도 표기)
    layer5_y = ly0 - 4 * (lh + gap) + lh / 2
    _arrow(ax, lx + lw, layer5_y, kx, ky0 + kh / 2 - 0.02,
           color=C_ACCENT_1, lw=2.0)
    sig_mid_x = (lx + lw + kx) / 2
    sig_mid_y = (layer5_y + ky0 + kh / 2 - 0.02) / 2
    ax.text(sig_mid_x, sig_mid_y + 0.02, "Signals", ha="center", va="bottom",
            color=C_ACCENT_1, fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=C_BG,
                      edgecolor="none", alpha=0.9))

    # Feedback loop arrow (right bottom → left top)
    feedback_y = ky0 - 4 * (lh + gap) - 0.02
    ax.text(0.5, 0.07, "↺ 사이징 피드백 루프: 실현 P&L → 룰별 multiplier → 다음 진입 사이즈 조정",
            ha="center", color=C_ACCENT_3, fontsize=11, fontweight="bold")

    # Title labels for two columns (위쪽 데이터소스 박스와 레이어 박스 사이 영역에 표기)
    ax.text(lx + lw / 2, 0.865, "분석 레이어 (Inference)",
            ha="center", color=C_ACCENT_2, fontsize=10, fontweight="bold")
    ax.text(kx + kw / 2, 0.865, "Kinetic Layer (Action + Learning)",
            ha="center", color=C_ACCENT_3, fontsize=10, fontweight="bold")

    _footer(fig, n, total)
    pdf.savefig(fig, facecolor=C_BG); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Page 3 — Layer 1 GARCH
# ──────────────────────────────────────────────────────────────

def page_garch(pdf, n, total):
    fig = _new_page()
    _title(fig, "Layer 1 — GARCH(1,1)",
           "조건부 변동성 + VRP(Volatility Risk Premium): IV가 RV보다 높을 때 헤지 비용 과다 → 변동성 매도 기회")
    # 좌측: 수식·해석
    ax_text = fig.add_axes((0.04, 0.10, 0.45, 0.74))
    ax_text.set_xlim(0, 1); ax_text.set_ylim(0, 1)
    ax_text.set_facecolor(C_BG)
    for s in ax_text.spines.values(): s.set_visible(False)
    ax_text.set_xticks([]); ax_text.set_yticks([])

    ax_text.text(0.0, 0.95, "수식", color=C_ACCENT_4, fontsize=13, fontweight="bold")
    ax_text.text(0.0, 0.88, "σ²_t = ω + α·ε²_(t-1) + β·σ²_(t-1)",
                 color=C_TEXT, fontsize=14, fontfamily="monospace")
    ax_text.text(0.0, 0.82, "VRP_true = ATM_IV - RV_20d   ← Alpaca 옵션 IV (2026-06-08~)",
                 color=C_TEXT, fontsize=11.5, fontfamily="monospace")

    ax_text.text(0.0, 0.72, "용어", color=C_ACCENT_4, fontsize=13, fontweight="bold")
    glossary = [
        ("σ_t", "시점 t의 조건부 표준편차(변동성). 매일 갱신됨."),
        ("ω, α, β", "GARCH 파라미터. α+β<1 이면 평균회귀."),
        ("VRP", "Volatility Risk Premium — IV(내재변동성) 대비 RV(실현변동성) 초과분."),
        ("RV", "Realized Volatility — 과거 20일 수익률 표준편차."),
        ("IV", "Implied Volatility — 옵션 가격에 내포된 미래 변동성."),
    ]
    y = 0.66
    for k, v in glossary:
        ax_text.text(0.0, y, k, color=C_ACCENT_2, fontsize=11,
                     fontweight="bold", fontfamily="monospace")
        ax_text.text(0.18, y, v, color=C_TEXT, fontsize=10)
        y -= 0.055

    ax_text.text(0.0, 0.32, "의사결정 시사 (VRP_true 기준)", color=C_ACCENT_4,
                 fontsize=13, fontweight="bold")
    sigs = [
        ("VRP_true > +2%", "옵션 IV > 실현 RV → STRADDLE ×0.5 페널티 (vol_overpriced)"),
        ("VRP_true ≈ 0",   "균형 — 중립"),
        ("VRP_true < -5%", "IV 과소평가 → 꼬리 리스크 경고"),
    ]
    y = 0.25
    for k, v in sigs:
        ax_text.text(0.0, y, "▸ " + k, color=C_ACCENT_4, fontsize=11,
                     fontweight="bold")
        ax_text.text(0.25, y, v, color=C_TEXT, fontsize=10)
        y -= 0.05

    # 우측: 실제 σ 시계열 차트
    ax_chart = fig.add_axes((0.54, 0.18, 0.42, 0.62))
    ax_chart.set_facecolor("#0f172a")
    try:
        import pandas as pd
        v = pd.read_parquet(OUTPUT_DIR / "garch_vol.parquet")
        for col, color in [("XLK", C_ACCENT_2), ("XLE", C_ACCENT_5),
                           ("XLF", C_ACCENT_4)]:
            if col in v.columns:
                ax_chart.plot(v.index, v[col], label=col, color=color, lw=1.4)
        ax_chart.set_title("GARCH 연환산 변동성 (실제 캐시)", color=C_TEXT,
                           fontsize=11)
        ax_chart.set_ylabel("σ_annual", color=C_MUTED, fontsize=10)
        ax_chart.legend(loc="upper right", frameon=False,
                        labelcolor=C_TEXT, fontsize=9)
        for s in ax_chart.spines.values(): s.set_color(C_LIGHT)
        ax_chart.tick_params(colors=C_MUTED, labelsize=8)
        ax_chart.grid(True, color=C_LIGHT, alpha=0.3, lw=0.5)
    except Exception as e:
        ax_chart.text(0.5, 0.5, f"chart unavailable: {e}",
                      ha="center", va="center", color=C_MUTED, fontsize=9)

    _footer(fig, n, total)
    pdf.savefig(fig, facecolor=C_BG); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Page 4 — Layer 2 EVT/GPD
# ──────────────────────────────────────────────────────────────

def page_evt(pdf, n, total):
    fig = _new_page()
    _title(fig, "Layer 2 — EVT / GPD (Extreme Value Theory)",
           "정규분포가 과소평가하는 꼬리(Fat Tail)를 직접 모델링 — 위기 시 손실 규모 정량화")

    ax_text = fig.add_axes((0.04, 0.10, 0.42, 0.74))
    ax_text.set_xlim(0, 1); ax_text.set_ylim(0, 1)
    ax_text.set_facecolor(C_BG)
    for s in ax_text.spines.values(): s.set_visible(False)
    ax_text.set_xticks([]); ax_text.set_yticks([])

    ax_text.text(0.0, 0.97, "왜 정규분포로는 부족한가",
                 color=C_ACCENT_4, fontsize=13, fontweight="bold", va="top")
    ax_text.text(0.0, 0.90,
                 "금융 수익률은 정규분포보다 꼬리가 두껍다(leptokurtosis).\n"
                 "표준 VaR(99%)는 이 꼬리 질량을 과소평가 → 실제 위기에서 자본 부족.",
                 color=C_TEXT, fontsize=10, linespacing=1.5, va="top")

    ax_text.text(0.0, 0.78, "POT 방법", color=C_ACCENT_4, fontsize=13,
                 fontweight="bold", va="top")
    ax_text.text(0.0, 0.71,
                 "1. 손실 분포 상위 10% (= 임계값 u 초과분) 추출\n"
                 "2. 초과분에 GPD(Generalized Pareto Distribution) 적합\n"
                 "   F(y) = 1 - (1 + ξ·y/σ)^(-1/ξ)",
                 color=C_TEXT, fontsize=10, linespacing=1.5, va="top")

    ax_text.text(0.0, 0.48, "꼬리 지수 ξ 의 해석",
                 color=C_ACCENT_4, fontsize=13, fontweight="bold")
    xi_table = [
        ("ξ < 0", "얇은 꼬리 — 손실에 상한"),
        ("ξ = 0", "지수 꼬리 — 정규 비슷"),
        ("ξ > 0", "FAT TAIL — 극단 손실 빈번"),
        ("ξ > 0.5", "분산 무한대 — VaR 무의미"),
        ("ξ > 1.0", "평균조차 무한대"),
    ]
    y = 0.41
    for k, v in xi_table:
        color = C_ACCENT_5 if "FAT" in v or "무한" in v else C_TEXT
        ax_text.text(0.0, y, k, color=color, fontsize=11,
                     fontweight="bold", fontfamily="monospace")
        ax_text.text(0.18, y, v, color=color, fontsize=10)
        y -= 0.05

    ax_text.text(0.0, 0.08, "→ 룰 fat_tail_alert : ξ > 0.20 → HEDGE/MONITOR",
                 color=C_ACCENT_5, fontsize=10, fontweight="bold")

    # 우측: 분포 비교 — 실제 XLE 수익률 히스토그램 vs 정규
    ax_chart = fig.add_axes((0.52, 0.18, 0.44, 0.62))
    ax_chart.set_facecolor("#0f172a")
    try:
        import pandas as pd
        r = pd.read_parquet(OUTPUT_DIR / "sector_returns.parquet")["XLE"].dropna()
        bins = np.linspace(-0.06, 0.06, 60)
        ax_chart.hist(r, bins=bins, color=C_ACCENT_4, alpha=0.55,
                      density=True, label="XLE 일별 수익률")
        x = np.linspace(bins.min(), bins.max(), 200)
        mu, sd = r.mean(), r.std()
        normal_pdf = (1 / (sd * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x - mu) / sd) ** 2)
        ax_chart.plot(x, normal_pdf, color=C_ACCENT_5, lw=2,
                      label=f"정규(μ={mu:.4f}, σ={sd:.4f})")
        # 음의 꼬리 강조
        tail_x = x[x < mu - 2.5 * sd]
        if len(tail_x):
            ax_chart.axvspan(tail_x.min(), tail_x.max(),
                             color=C_ACCENT_5, alpha=0.1)
            ax_chart.text(tail_x.mean(), ax_chart.get_ylim()[1] * 0.7,
                          "← 정규 모델이\n   놓치는 영역",
                          ha="center", color=C_ACCENT_5, fontsize=9)
        ax_chart.set_title("XLE 일별 수익률 분포 vs 정규 (실제 데이터)",
                           color=C_TEXT, fontsize=11)
        ax_chart.set_xlabel("일별 로그수익률", color=C_MUTED, fontsize=10)
        ax_chart.legend(loc="upper right", frameon=False,
                        labelcolor=C_TEXT, fontsize=9)
        for s in ax_chart.spines.values(): s.set_color(C_LIGHT)
        ax_chart.tick_params(colors=C_MUTED, labelsize=8)
        ax_chart.grid(True, color=C_LIGHT, alpha=0.3, lw=0.5)
    except Exception as e:
        ax_chart.text(0.5, 0.5, f"chart unavailable: {e}",
                      ha="center", va="center", color=C_MUTED, fontsize=9)

    _footer(fig, n, total)
    pdf.savefig(fig, facecolor=C_BG); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Page 5 — Layer 3 Copula
# ──────────────────────────────────────────────────────────────

def page_copula(pdf, n, total):
    fig = _new_page()
    _title(fig, "Layer 3 — Copula 꼬리 의존성",
           "정상 상관계수는 위기 시 상승 — 평균 상관이 아니라 꼬리(5%) 만의 동시 발생률을 직접 측정")

    ax_text = fig.add_axes((0.04, 0.10, 0.42, 0.74))
    ax_text.set_xlim(0, 1); ax_text.set_ylim(0, 1)
    ax_text.set_facecolor(C_BG)
    for s in ax_text.spines.values(): s.set_visible(False)
    ax_text.set_xticks([]); ax_text.set_yticks([])

    ax_text.text(0.0, 0.95, "수식 (경험적 비모수)",
                 color=C_ACCENT_4, fontsize=13, fontweight="bold")
    ax_text.text(0.0, 0.86,
                 "λ_L(u) = P(X < Q_X(u) | Y < Q_Y(u))\n"
                 "         (u=0.05 → 양쪽 모두 하위 5% 분위에 동시 발생)",
                 color=C_TEXT, fontsize=10.5, fontfamily="monospace",
                 linespacing=1.5)
    ax_text.text(0.0, 0.74,
                 "λ_U(u) = P(X > Q_X(1-u) | Y > Q_Y(1-u))   (동시 급등)",
                 color=C_TEXT, fontsize=10.5, fontfamily="monospace")

    ax_text.text(0.0, 0.62, "용어", color=C_ACCENT_4, fontsize=13, fontweight="bold")
    g = [
        ("Copula", "주변분포와 의존구조를 분리해 모델링하는 함수족."),
        ("λ_L (Lower Tail Dep.)", "동시 폭락 확률. 0 = 무관, 1 = 항상 함께 폭락."),
        ("λ_U (Upper Tail Dep.)", "동시 급등 확률."),
        ("Q_X(u)",                "X 의 u 분위수 (예: u=0.05 → 하위 5%)."),
    ]
    y = 0.55
    for k, v in g:
        ax_text.text(0.0, y, k, color=C_ACCENT_2, fontsize=10.5,
                     fontweight="bold")
        ax_text.text(0.34, y, v, color=C_TEXT, fontsize=10)
        y -= 0.055

    ax_text.text(0.0, 0.30, "온톨로지 룰 연결",
                 color=C_ACCENT_4, fontsize=13, fontweight="bold", va="top")
    ax_text.text(0.0, 0.24,
                 "co_crash_cluster:\n"
                 "  λ_L ≥ 0.40 인 피어 섹터가 3개 이상 → HEDGE\n"
                 "  의미: '분산투자가 위기 때 안 듣는' 시스템 위험 증폭자",
                 color=C_TEXT, fontsize=10, linespacing=1.6, va="top")
    ax_text.text(0.0, 0.04, "↔ Layer 4 (Cholesky) 는 평상시(피어슨)를, Layer 3 는 꼬리를 — 직교 분업",
                 color=C_ACCENT_3, fontsize=9.5, style="italic")

    # 우측: 실제 lambda_L heatmap
    ax_chart = fig.add_axes((0.52, 0.16, 0.44, 0.64))
    try:
        import pandas as pd
        lower = pd.read_csv(OUTPUT_DIR / "tail_dep_sector_lower.csv", index_col=0)
        # 11×11 매트릭스에서 NaN → 대각 0
        M = lower.values.astype(float)
        np.fill_diagonal(M, 0)
        # 대칭화 (CSV가 상삼각인 경우)
        M = np.where(np.isnan(M), M.T, M)
        np.fill_diagonal(M, np.nan)

        im = ax_chart.imshow(M, cmap="magma", vmin=0, vmax=0.6, aspect="auto")
        ax_chart.set_xticks(range(len(lower.columns)))
        ax_chart.set_yticks(range(len(lower.index)))
        ax_chart.set_xticklabels(lower.columns, rotation=45, ha="right",
                                 color=C_MUTED, fontsize=8)
        ax_chart.set_yticklabels(lower.index, color=C_MUTED, fontsize=8)
        ax_chart.set_title("섹터 × 섹터 λ_L (실제 캐시) — 진할수록 위기 동조",
                           color=C_TEXT, fontsize=11)
        cbar = fig.colorbar(im, ax=ax_chart, shrink=0.85)
        cbar.ax.tick_params(colors=C_MUTED, labelsize=8)
        cbar.outline.set_edgecolor(C_LIGHT)
    except Exception as e:
        ax_chart.set_facecolor("#0f172a")
        ax_chart.text(0.5, 0.5, f"chart unavailable: {e}",
                      ha="center", va="center", color=C_MUTED, fontsize=9)

    _footer(fig, n, total)
    pdf.savefig(fig, facecolor=C_BG); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Page 6 — Layer 4 Cholesky
# ──────────────────────────────────────────────────────────────

def page_cholesky(pdf, n, total, live):
    fig = _new_page()
    _title(fig, "Layer 4 — Multivariate MC + Cholesky 분산 분해 (NEW 2026-06-08)",
           "평상시 가우시안 의존성. '각 섹터 분산이 어느 *직교* 충격에서 왔는가' 를 정량화")

    ax_text = fig.add_axes((0.04, 0.10, 0.42, 0.74))
    ax_text.set_xlim(0, 1); ax_text.set_ylim(0, 1)
    ax_text.set_facecolor(C_BG)
    for s in ax_text.spines.values(): s.set_visible(False)
    ax_text.set_xticks([]); ax_text.set_yticks([])

    ax_text.text(0.0, 0.97, "수식", color=C_ACCENT_4, fontsize=13, fontweight="bold")
    ax_text.text(0.0, 0.91, "Σ = L · L^T   (L = 하삼각 콜레스키 인자)",
                 color=C_TEXT, fontsize=10.5, fontfamily="monospace")
    ax_text.text(0.0, 0.86,
                 "var_attr[i,j] = L[i,j]² / Σ_k L[i,k]²  (행합=1)",
                 color=C_TEXT, fontsize=10.5, fontfamily="monospace")

    ax_text.text(0.0, 0.77, "변수 순서 = 가정 (인과 아님)",
                 color=C_ACCENT_5, fontsize=13, fontweight="bold", va="top")
    ax_text.text(0.0, 0.70,
                 "PCMCI 는 매크로에만 적용 가능 - 섹터엔 인과 가설 없음.\n"
                 "대신 |corr| 가중 eigenvector centrality 내림차순(휴리스틱).\n"
                 "centrality 1위 = 첫 컬럼 = 'shock origin prior' (발견 X, 가정 O).\n"
                 "순서 바꾸면 var_attr 도 바뀜 — SVAR 재귀식별의 알려진 성질.",
                 color=C_TEXT, fontsize=10, linespacing=1.5, va="top")

    ax_text.text(0.0, 0.50, "노드 속성", color=C_ACCENT_4, fontsize=13,
                 fontweight="bold")
    attr = [
        ("var_decomposition", "j 직교 충격이 자기 분산에 기여한 비율 (합=1)"),
        ("self_share",        "= var_decomp[self]. ≥0.70 → variance_concentrated"),
        ("propagation_score", "자기 충격이 다른 섹터 분산에 기여 합. ≥0.50 → shock_prop"),
        ("cholesky_order",    "0 = 가장 시스템적인 충격원"),
    ]
    y = 0.43
    for k, v in attr:
        ax_text.text(0.0, y, k, color=C_ACCENT_2, fontsize=10,
                     fontweight="bold", fontfamily="monospace")
        ax_text.text(0.0, y - 0.03, v, color=C_TEXT, fontsize=9.5)
        y -= 0.075

    ax_text.text(0.0, 0.10, "엣지 CO_MOVES_WITH (j → i, 비대칭)",
                 color=C_ACCENT_4, fontsize=11, fontweight="bold")
    ax_text.text(0.0, 0.04,
                 "λ_L (Layer 3, 꼬리) 와 var_attr (Layer 4, 본체) 가 직교 공존",
                 color=C_TEXT, fontsize=9.5, style="italic")

    # 우측: 실제 XLK 분산 분해 막대 (live data)
    ax_chart = fig.add_axes((0.52, 0.16, 0.44, 0.64))
    ax_chart.set_facecolor("#0f172a")
    try:
        by_sector = live["variance_decomposition"]["by_sector"]
        # XLK 행
        target = "XLK"
        row = next(r for r in by_sector if r["ticker"] == target)
        # 차트 데이터: self + 외부 상위 3 + others
        labels = [f"{target} (self)"]
        values = [row["self_share"]]
        colors = [C_LIGHT]
        for i, src in enumerate(row["top_sources"]):
            labels.append(src["src"])
            values.append(src["share"])
            colors.append([C_ACCENT_1, C_ACCENT_4, C_ACCENT_2][i % 3])
        used = sum(values)
        if used < 0.999:
            labels.append("기타")
            values.append(1 - used)
            colors.append("#1f2937")

        # 누적 가로 막대
        left = 0
        for lab, val, col in zip(labels, values, colors):
            ax_chart.barh(0, val, left=left, color=col, edgecolor=C_BG, lw=1)
            if val > 0.04:
                ax_chart.text(left + val / 2, 0,
                              f"{lab}\n{val*100:.1f}%", ha="center",
                              va="center", color=C_TEXT, fontsize=10,
                              fontweight="bold")
            left += val
        ax_chart.set_xlim(0, 1); ax_chart.set_ylim(-0.6, 1.0)
        ax_chart.set_yticks([])
        ax_chart.set_title(
            f"{target} 분산 분해 (실제 산출) — "
            f"order #{row['cholesky_order']}, prop={row['propagation_score']:.2f}",
            color=C_TEXT, fontsize=11)
        ax_chart.set_xlabel("분산 기여 비율 (합 = 1.0)", color=C_MUTED, fontsize=10)
        ax_chart.tick_params(colors=C_MUTED, labelsize=8)
        for s in ax_chart.spines.values(): s.set_color(C_LIGHT)

        # 해석 텍스트
        ax_chart.text(0.5, 0.7,
                      f"{target} 의 평상시 변동 중 {row['self_share']*100:.0f}% 는 자기 직교 충격,\n"
                      f"{row['top_sources'][0]['src']} 시장 충격이 {row['top_sources'][0]['share']*100:.0f}% 기여",
                      ha="center", color=C_ACCENT_3, fontsize=10.5,
                      transform=ax_chart.transAxes)
    except Exception as e:
        ax_chart.text(0.5, 0.5, f"chart unavailable: {e}",
                      ha="center", va="center", color=C_MUTED, fontsize=9)

    _footer(fig, n, total)
    pdf.savefig(fig, facecolor=C_BG); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Page 7 — PCMCI causal chain
# ──────────────────────────────────────────────────────────────

def page_pcmci(pdf, n, total):
    fig = _new_page()
    _title(fig, "PCMCI — 거시 변수 인과 발견",
           "이론에 의존하지 않고 데이터에서 직접 변수 간 인과 체인 추출 (Granger보다 보수적)")

    ax_text = fig.add_axes((0.04, 0.10, 0.36, 0.74))
    ax_text.set_xlim(0, 1); ax_text.set_ylim(0, 1)
    ax_text.set_facecolor(C_BG)
    for s in ax_text.spines.values(): s.set_visible(False)
    ax_text.set_xticks([]); ax_text.set_yticks([])

    ax_text.text(0.0, 0.97, "왜 PCMCI", color=C_ACCENT_4, fontsize=13,
                 fontweight="bold", va="top")
    ax_text.text(0.0, 0.91,
                 "Granger 인과: 다른 변수들의 공통 원인을 통제 안 함\n"
                 "→ 허위 인과(spurious causality) 발생.\n\n"
                 "PCMCI = PC + MCI:\n"
                 " · PC: 각 변수의 과거값 부모를 독립적으로 선별\n"
                 " · MCI: X→Y 검정 시 X·Y 과거 + 공통원인 모두 조건부 통제",
                 color=C_TEXT, fontsize=10, linespacing=1.5, va="top")

    ax_text.text(0.0, 0.50, "용어", color=C_ACCENT_4, fontsize=13, fontweight="bold")
    g = [
        ("lag_months",     "원인 → 결과 까지의 지연(개월)"),
        ("stability",      "structural / emerging / historical"),
        ("structural",     "전체 + 현재 레짐 양쪽 유의 — 최고 신뢰"),
        ("emerging",       "현재 레짐에서만 유의 — 2022 이후 신규"),
        ("change-point",   "공분산 행렬 Frobenius 거리 피크 = 레짐 전환점"),
    ]
    y = 0.43
    for k, v in g:
        ax_text.text(0.0, y, k, color=C_ACCENT_2, fontsize=10,
                     fontweight="bold", fontfamily="monospace")
        ax_text.text(0.30, y, v, color=C_TEXT, fontsize=10)
        y -= 0.055

    ax_text.text(0.0, 0.10,
                 "현재 레짐 전환점: 2022-02 (우크라이나 + Fed 긴축 시작)",
                 color=C_ACCENT_3, fontsize=10, fontweight="bold")
    ax_text.text(0.0, 0.05,
                 "총 42개 링크 발견 (structural 5 / emerging 22 / historical 15)",
                 color=C_MUTED, fontsize=10)

    # 우측: 인과 체인 다이어그램
    ax_g = fig.add_axes((0.43, 0.10, 0.54, 0.74))
    ax_g.set_xlim(0, 1); ax_g.set_ylim(0, 1)
    ax_g.set_facecolor(C_BG)
    for s in ax_g.spines.values(): s.set_visible(False)
    ax_g.set_xticks([]); ax_g.set_yticks([])

    nodes = {
        "OIL":           (0.10, 0.85),
        "CPI":           (0.30, 0.85),
        "CREDIT_SPREAD": (0.10, 0.45),
        "US10Y":         (0.50, 0.85),
        "US2Y":          (0.50, 0.60),
        "FED_FUNDS":     (0.70, 0.60),
        "VIX":           (0.88, 0.45),
        "DXY":           (0.30, 0.20),
    }
    for name, (x, y) in nodes.items():
        c = "#fbbf24" if name in ("VIX",) else C_ACCENT_2
        p = FancyBboxPatch((x - 0.06, y - 0.04), 0.12, 0.08,
                           boxstyle="round,pad=0.005,rounding_size=0.02",
                           linewidth=1.4, edgecolor=c, facecolor="#111827")
        ax_g.add_patch(p)
        ax_g.text(x, y, name, ha="center", va="center", color=C_TEXT,
                  fontsize=10, fontweight="bold")

    edges = [
        ("OIL", "CPI", "+1m", "stru", C_ACCENT_3),
        ("CPI", "US2Y", "+3m", "emer", C_ACCENT_4),
        ("CPI", "US10Y", "+2m", "stru", C_ACCENT_3),
        ("US2Y", "FED_FUNDS", "+1m", "stru", C_ACCENT_3),
        ("FED_FUNDS", "VIX", "-3m", "stru", C_ACCENT_3),
        ("FED_FUNDS", "CREDIT_SPREAD", "+1m", "stru", C_ACCENT_3),
        ("CREDIT_SPREAD", "VIX", "+1m", "emer", C_ACCENT_4),
        ("US10Y", "DXY", "-2m", "emer", C_ACCENT_4),
    ]
    # 라벨이 교차 화살표 위에서 겹치는 것을 막기 위해 (src,dst) 별로 명시적
    # 라벨 위치(midpoint 에서 노멀 방향 ±offset)를 지정. 배경 박스로 다른 화살표와 분리.
    label_offsets = {
        ("OIL", "CPI"):              ( 0.00,  0.040),
        ("CPI", "US2Y"):             ( 0.075, 0.000),  # 대각선 우측
        ("CPI", "US10Y"):            ( 0.00,  0.040),
        ("US2Y", "FED_FUNDS"):       ( 0.00, -0.045),  # 박스 위 충돌 → 아래로
        ("FED_FUNDS", "VIX"):        ( 0.045, 0.018),
        ("FED_FUNDS", "CREDIT_SPREAD"): (0.00, 0.050),
        ("CREDIT_SPREAD", "VIX"):    ( 0.00,  0.030),
        ("US10Y", "DXY"):            (-0.080, 0.000),  # 좌측 — US2Y 라벨과 분리
    }
    for src, dst, lag, stab, color in edges:
        x1, y1 = nodes[src]; x2, y2 = nodes[dst]
        dx, dy = x2 - x1, y2 - y1
        L = np.hypot(dx, dy)
        ox, oy = dx / L * 0.06, dy / L * 0.04
        _arrow(ax_g, x1 + ox, y1 + oy, x2 - ox, y2 - oy, color=color, lw=1.4)
        lo = label_offsets.get((src, dst), (0.0, 0.020))
        lx = x1 + dx / 2 + lo[0]
        ly = y1 + dy / 2 + lo[1]
        ax_g.text(lx, ly, f"{lag}·{stab}", ha="center", va="center",
                  color=color, fontsize=8,
                  bbox=dict(boxstyle="round,pad=0.15", facecolor=C_BG,
                            edgecolor="none", alpha=0.9))

    ax_g.text(0.5, 0.04,
              "룰 causal_chain_monitor 는 2026-06-10 은퇴 (4개월 stale·변별력 낮음) — "
              "PCMCI 는 TRANSMITS_TO 엣지(UI 분석)로만 유지",
              ha="center", color=C_MUTED, fontsize=9, style="italic")
    ax_g.text(0.5, 0.005,
              "stru = structural, emer = emerging  ·  일별 교란 통제는 다변량 partial 회귀가 담당 (다음 페이지)",
              ha="center", color=C_MUTED, fontsize=8)

    _footer(fig, n, total)
    pdf.savefig(fig, facecolor=C_BG); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Page 8 — Layer 5 Ontology Gate (rules table)
# ──────────────────────────────────────────────────────────────

def page_rules(pdf, n, total):
    fig = _new_page(figsize=(12.0, 9.0))
    _title(fig, "Layer 5 — Ontology Gate (9 추론 룰, 2026-06-16)",
           "11→8(R5 정합성 의심 3제거)→9. R9: event_vol 신설 — 실현변동성 급등→롱 볼 (VIX 비대칭 보완).")

    ax = _blank_axes(fig, (0.04, 0.06, 0.92, 0.80))

    rules = [
        # (rule, condition, signal, meaning, is_new, is_removed)
        ("natural_hedge",          "gamma(VIX) > 0.10",                          "OVERWEIGHT", "VIX 급등 시 볼록 수익", False, False),
        ("crash_vulnerable",       "delta(VIX) < -0.008 & tail_asymm < -0.001",  "UNDERWEIGHT", "하락장 비대칭 손실", False, False),
        ("co_crash_cluster",       "λ_L ≥ 0.40 peer ≥ 3개",                       "HEDGE",       "위기 시 분산 효과 소멸 ✓ +20.4% 백테스트", False, False),
        ("rate_beneficiary",       "delta_ctrl > 0.002 & FDR q<0.10",             "OVERWEIGHT", "금리 수혜 — 유효 섹터는 상태표*가 단일 진실원", False, False),
        ("rate_victim",            "delta_ctrl < -0.001 & FDR q<0.10",            "UNDERWEIGHT", "금리 피해 — 유효 섹터는 상태표* 참조", False, False),
        ("vol_overpriced",         "VRP_true > 2% (Alpaca IV)",                   "OVERWEIGHT", "IV>RV — directional 제외 + SHORT 페어", False, False),
        ("event_vol ★R9",          "realized z ≥ 2.5 & VRP_iv ≤ 0",               "→ 롱 볼",      "VIX 비대칭 보완: 실현변동 급등+싼IV → 롱스트래들", True, False),
        ("fat_tail_alert",         "ξ (GPD) > 0.20",                              "HEDGE/MONI",  "정규 VaR 부적합 ✓ +9.4% 백테스트", False, False),
        ("normal_var_inadequate",  "CF VaR / Normal VaR > 2.5",                   "MONITOR",     "정규 VaR 실제의 절반", False, False),
        ("thin_tail_greenlight 💤", "CI 95%상한: ξ<0.10 & λ_L<0.20 & CF<1.5",      "MONITOR",     "死코드 — 발화 불가 (↓ 하단 주)", True, False),
        ("─" * 30, "", "", "── 2026-06-10 제거 (정합성 의심) ──", False, False),
        ("~~causal_chain_monitor~~", "PCMCI 4개월 stale, 변별력 낮음",                "—",           "백테스트 -$649 / n=50", False, True),
        ("~~shock_propagator~~",     "Shapley 평균이라도 비인과 휴리스틱",             "—",           "백테스트 -$1,949 / n=50 (최대 손실)", False, True),
        ("~~variance_concentrated~~", "동일 — 상관 해석만",                           "—",           "비인과 제거", False, True),
    ]

    # Header
    headers = ["룰명", "조건", "시그널", "의미"]
    cols_x = [0.02, 0.27, 0.55, 0.68]

    y = 0.96
    for h, x in zip(headers, cols_x):
        ax.text(x, y, h, color=C_ACCENT_4, fontsize=11, fontweight="bold")

    y -= 0.035
    for rule in rules:
        rule_name, cond, sig, meaning, is_new, is_removed = rule
        ax.axhline(y + 0.015, xmin=0.02, xmax=0.98, color=C_LIGHT, lw=0.4, alpha=0.3)
        if is_new:
            color = C_ACCENT_3      # emerald
        elif is_removed:
            color = "#6b7280"       # gray (removed)
        else:
            color = C_TEXT
        for txt, x in zip((rule_name, cond, sig, meaning), cols_x):
            ax.text(x, y, txt, color=color, fontsize=9.5,
                    fontweight="bold" if x == cols_x[0] else "normal")
        y -= 0.0415

    # 💤 thin_tail_greenlight 휴면 사유 전면화 (R8 검토 3) — 셀에 안 잘리게 본문에
    ax.text(0.02, 0.35,
            "[휴면] thin_tail_greenlight (무한손실 숏볼 게이트 — ξ·λ_L·CF 95% 상한으로만 발화): "
            "P2-5 실측 — 게이트 통과에 ξ≈10.7년·λ_L≈1.9년치 표본 필요 = 현 데이터 체계에서\n"
            "수학적으로 발화 불가. 횡단면 상대임계 재설계 없이는 死코드(영구 동결) — 흐리지 않고 명시.",
            color="#6ee7b7", fontsize=8.5, linespacing=1.4, va="top")

    # 레짐별 활성 룰 (9개)
    ax.text(0.02, 0.255, "레짐별 활성 룰 (9개, 2026-06-16) — event_vol 은 레짐 독립(전 레짐)",
            color=C_ACCENT_4, fontsize=12, fontweight="bold")
    regimes = [
        ("low_vix (VIX < 15)",
         "vol_overpriced · rate_beneficiary · thin_tail_greenlight · event_vol"),
        ("mid_vix (15 ≤ VIX < 25)",
         "rate_* · normal_var_inadequate · vol_overpriced · thin_tail_greenlight · event_vol"),
        ("high_vix (VIX ≥ 25)",
         "natural_hedge · crash_vulnerable · co_crash_cluster · fat_tail_alert · rate_victim · event_vol"),
    ]
    y = 0.205
    for k, v in regimes:
        ax.text(0.02, y, k, color=C_ACCENT_2, fontsize=10, fontweight="bold")
        ax.text(0.26, y, v, color=C_TEXT, fontsize=9)
        y -= 0.038

    ax.text(0.02, 0.05,
            "* 단일 진실원 = output/rule_sector_state.json (매 사이클 갱신, FDR q<0.10 데이터 유효성). "
            "문서의 섹터 명단이 아니라 이 파일이 룰 발화와 '시그널은퇴' 청산의 기준.",
            color=C_ACCENT_4, fontsize=9, fontweight="bold")

    _footer(fig, n, total)
    pdf.savefig(fig, facecolor=C_BG); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Page — Sensitivity Audit: raw vs controlled delta (NEW 2026-06-12)
# ──────────────────────────────────────────────────────────────

def page_sensitivity(pdf, n, total, live):
    fig = _new_page(figsize=(12.0, 9.0))
    _title(fig, "민감도 감사 — 이변량(raw) vs 다변량 통제(ctrl) delta (NEW 2026-06-12)",
           "R_s = α + Σ δ_j·ΔX_j + Σ γ_j·ΔX_j² + β_mkt·R_SPY — OIL 포함 전 매크로 동시 + 시장 통제 (FF MKT 역할)")

    # 좌측: 문제와 해법
    ax_text = fig.add_axes((0.04, 0.08, 0.34, 0.76))
    ax_text.set_xlim(0, 1); ax_text.set_ylim(0, 1)
    ax_text.set_facecolor(C_BG)
    for s in ax_text.spines.values(): s.set_visible(False)
    ax_text.set_xticks([]); ax_text.set_yticks([])

    ax_text.text(0.0, 0.98, "문제 — 누락변수 교란", color=C_ACCENT_5,
                 fontsize=13, fontweight="bold", va="top")
    ax_text.text(0.0, 0.91,
                 "이변량 회귀(섹터 ~ 매크로 1개씩)는 같은 표본에서\n"
                 "동시에 움직인 제3변수를 통제하지 못한다.\n"
                 "호르무즈 표본: 유가↑ → 금리↑ 동시 발생\n"
                 "→ OIL 을 뺀 delta(US10Y) 가 유가 효과를 흡수\n"
                 "→ XLE 가 가짜 '금리 수혜' 로 분류됨.",
                 color=C_TEXT, fontsize=9.5, linespacing=1.5, va="top")

    ax_text.text(0.0, 0.62, "해법 — partial delta", color=C_ACCENT_3,
                 fontsize=13, fontweight="bold", va="top")
    ax_text.text(0.0, 0.55,
                 "전 매크로(VIX·US10Y·US2Y·DXY·OIL) 동시 투입\n"
                 "+ SPY 수익률 통제 (시장베타 = FF MKT 팩터 통제).\n"
                 "PCMCI ParCorr 의 조건부 독립 아이디어를\n"
                 "일별 주기(~360 표본)로 가져온 것.\n"
                 "rate_* 룰은 delta_ctrl 로만 발화 (06-12~).",
                 color=C_TEXT, fontsize=9.5, linespacing=1.5, va="top")

    ax_text.text(0.0, 0.27, "verdict", color=C_ACCENT_4, fontsize=13,
                 fontweight="bold", va="top")
    verdicts = [
        ("KILLED",    "raw 만 유의 — 교란이었음 (룰 차단)", C_ACCENT_5),
        ("FLIPPED",   "부호 반전 — 심한 교란",              C_ACCENT_5),
        ("EMERGED",   "ctrl 만 유의 — 가려져 있던 민감도",   C_ACCENT_2),
        ("CONFIRMED", "양쪽 생존 — 진짜 민감도",             C_ACCENT_3),
    ]
    y = 0.21
    for k, v, c in verdicts:
        ax_text.text(0.0, y, k, color=c, fontsize=9.5, fontweight="bold",
                     fontfamily="monospace")
        ax_text.text(0.33, y, v, color=C_TEXT, fontsize=9)
        y -= 0.045

    # 우측: 라이브 감사 테이블 (ontology_signals.json sensitivity_audit)
    ax_t = fig.add_axes((0.42, 0.08, 0.55, 0.76))
    ax_t.set_xlim(0, 1); ax_t.set_ylim(0, 1)
    ax_t.set_facecolor(C_BG)
    for s in ax_t.spines.values(): s.set_visible(False)
    ax_t.set_xticks([]); ax_t.set_yticks([])

    audit = (live.get("sensitivity_audit") or [])[:16]
    headers = [("Sector × Macro", 0.00), ("δ raw (t)", 0.30),
               ("δ ctrl (t)", 0.55), ("Verdict", 0.82)]
    y = 0.98
    for h, x in headers:
        ax_t.text(x, y, h, color=C_ACCENT_4, fontsize=10, fontweight="bold")
    y -= 0.045
    v_color = {"killed": C_ACCENT_5, "flipped": C_ACCENT_5,
               "emerged": C_ACCENT_2, "confirmed": C_ACCENT_3}
    if not audit:
        ax_t.text(0.5, 0.5, "sensitivity_audit 데이터 없음 — trigger.py 먼저 실행",
                  ha="center", color=C_MUTED, fontsize=10)
    for r in audit:
        ax_t.axhline(y + 0.018, xmin=0.0, xmax=1.0, color=C_LIGHT, lw=0.4, alpha=0.3)
        c = v_color.get(r["verdict"], C_TEXT)
        ax_t.text(0.00, y, f"{r['sector']} × {r['macro']}", color=C_TEXT,
                  fontsize=9, fontweight="bold", fontfamily="monospace")
        d_raw = f"{r['delta_raw']:+.4f} ({r['t_raw']:+.1f})" if r.get("delta_raw") is not None else "—"
        d_ctl = f"{r['delta_ctrl']:+.4f} ({r['t_ctrl']:+.1f})" if r.get("delta_ctrl") is not None else "—"
        ax_t.text(0.30, y, d_raw, color=C_MUTED, fontsize=8.5, fontfamily="monospace")
        ax_t.text(0.55, y, d_ctl, color=C_TEXT, fontsize=8.5, fontfamily="monospace")
        ax_t.text(0.82, y, r["verdict"].upper(), color=c, fontsize=8.5,
                  fontweight="bold", fontfamily="monospace")
        y -= 0.052

    # 하단: 핵심 발견 — 섹터 명단은 적지 않는다 (단일 진실원 = rule_sector_state.json,
    # 위 표가 그 라이브 출력. 문서에 명단을 박으면 p.8/9/14 가 서로 어긋난다)
    fig.text(0.04, 0.045,
             "핵심 발견: XLE '금리 수혜' = 유가 교란 (직교화 후에도 t=0.9) · XLK×VIX FLIPPED = 시장베타 · "
             "직교화+FDR 로 일부 rate_victim 도 통제 후 유의성 소멸로 사망 — 현재 유효 명단은 위 표(=상태표)가 유일한 기준",
             color=C_ACCENT_4, fontsize=9.5, fontweight="bold")

    _footer(fig, n, total)
    pdf.savefig(fig, facecolor=C_BG); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Page 9 — NORTA Joint drawdown (live data)
# ──────────────────────────────────────────────────────────────

def page_joint(pdf, n, total, live):
    fig = _new_page()
    _title(fig, "NORTA Joint Drawdown — 시장 구조 base rate",
           "시그널과 무관하게 시장 상관/변동성 구조만으로 본 21영업일 동시 폭락 확률")

    ax_text = fig.add_axes((0.04, 0.10, 0.42, 0.74))
    ax_text.set_xlim(0, 1); ax_text.set_ylim(0, 1)
    ax_text.set_facecolor(C_BG)
    for s in ax_text.spines.values(): s.set_visible(False)
    ax_text.set_xticks([]); ax_text.set_yticks([])

    ax_text.text(0.0, 0.97, "NORTA 절차", color=C_ACCENT_4, fontsize=13,
                 fontweight="bold", va="top")
    ax_text.text(0.0, 0.90,
                 "1. z ~ N(0, I) 독립 표준정규 샘플\n"
                 "2. g = z · L^T  - Cholesky 로 상관 주입\n"
                 "3. r_t = μ_d + g · σ_d  - 섹터별 변동성 스케일\n"
                 "4. cum_t = Σ r_t  - 누적 로그수익률\n"
                 "5. drawdown = min over horizon",
                 color=C_TEXT, fontsize=10, fontfamily="monospace",
                 linespacing=1.5, va="top")

    ax_text.text(0.0, 0.56, "용어", color=C_ACCENT_4, fontsize=13, fontweight="bold")
    g = [
        ("NORTA",     "NORmal-To-Anything — 가우시안 의존성+자유 마진"),
        ("σ_d",       "섹터별 일별 변동성 (GARCH 연환산 / √252)"),
        ("τ (tau)",   "drawdown 임계값 (예: 5% 손실)"),
        ("P(≥절반)",  "전체 11개 섹터 중 절반 이상 동시 hit 확률"),
        ("base rate", "시그널 없이도 그냥 깔려 있는 위험도"),
    ]
    y = 0.49
    for k, v in g:
        ax_text.text(0.0, y, k, color=C_ACCENT_2, fontsize=10.5,
                     fontweight="bold")
        ax_text.text(0.22, y, v, color=C_TEXT, fontsize=10)
        y -= 0.055

    ax_text.text(0.0, 0.18, "의의", color=C_ACCENT_4, fontsize=13, fontweight="bold")
    ax_text.text(0.0, 0.14,
                 "이 숫자가 '아무 시그널 없어도 시장이 줄 위험량'.\n"
                 "듀얼 전략 시그널의 P&L 기댓값 평가 시 기준선으로 사용.\n"
                 "시그널이 알파(α)를 만든다면 base rate 위로 P&L 분포가 이동해야 함.",
                 color=C_TEXT, fontsize=10, linespacing=1.5, va="top")

    # 우측: per_threshold 차트 (실제 시뮬값)
    ax_chart = fig.add_axes((0.52, 0.16, 0.44, 0.64))
    ax_chart.set_facecolor("#0f172a")
    try:
        js = live["joint_simulation"]
        rows = js["per_threshold"]
        taus = [r["threshold"] * 100 for r in rows]
        p_any = [r["p_any"] * 100 for r in rows]
        p_half = [r["p_half"] * 100 for r in rows]
        p_all = [r["p_all"] * 100 for r in rows]

        x = np.arange(len(taus))
        w = 0.25
        ax_chart.bar(x - w, p_any, w, color=C_ACCENT_3, label="P(≥1 hit)")
        ax_chart.bar(x, p_half, w, color=C_ACCENT_4, label="P(≥절반 hit)")
        ax_chart.bar(x + w, p_all, w, color=C_ACCENT_5, label="P(전부 hit)")
        for i, (a, h, l) in enumerate(zip(p_any, p_half, p_all)):
            ax_chart.text(i - w, a + 1.5, f"{a:.1f}", ha="center",
                          color=C_TEXT, fontsize=9)
            ax_chart.text(i, h + 1.5, f"{h:.1f}", ha="center",
                          color=C_TEXT, fontsize=9)
            ax_chart.text(i + w, l + 1.5, f"{l:.2f}", ha="center",
                          color=C_TEXT, fontsize=9)
        ax_chart.set_xticks(x)
        ax_chart.set_xticklabels([f"τ={t:.1f}%" for t in taus],
                                 color=C_MUTED, fontsize=10)
        ax_chart.set_ylabel("확률 (%)", color=C_MUTED, fontsize=10)
        ax_chart.set_title(
            f"Joint Drawdown (horizon={js['horizon']}d, n_sim={js['n_sim']:,})",
            color=C_TEXT, fontsize=11)
        ax_chart.legend(loc="upper right", frameon=False,
                        labelcolor=C_TEXT, fontsize=9)
        ax_chart.tick_params(colors=C_MUTED, labelsize=8)
        for s in ax_chart.spines.values(): s.set_color(C_LIGHT)
        ax_chart.grid(True, axis="y", color=C_LIGHT, alpha=0.3, lw=0.5)
        ax_chart.set_ylim(0, 110)
    except Exception as e:
        ax_chart.text(0.5, 0.5, f"chart unavailable: {e}",
                      ha="center", va="center", color=C_MUTED, fontsize=9)

    _footer(fig, n, total)
    pdf.savefig(fig, facecolor=C_BG); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Page 10 — Dual strategy
# ──────────────────────────────────────────────────────────────

def page_dual(pdf, n, total):
    fig = _new_page(figsize=(12.0, 9.0))
    _title(fig, "트리플 전략 — 같은 시그널, 세 가지 베팅 (R9, 2026-06-16)",
           "LONG STRADDLE (위기 룰 + event_vol) + DIRECTIONAL (rate beta) + SHORT STRADDLE (그린라이트)")

    ax = _blank_axes(fig, (0.04, 0.06, 0.92, 0.82))

    # Central signal node
    sig_x, sig_y = 0.40, 0.70
    sig_w, sig_h = 0.20, 0.14
    p = FancyBboxPatch((sig_x, sig_y), sig_w, sig_h,
                       boxstyle="round,pad=0.02,rounding_size=0.04",
                       linewidth=2.0, edgecolor=C_ACCENT_1, facecolor=C_PANEL)
    ax.add_patch(p)
    ax.text(sig_x + sig_w / 2, sig_y + sig_h - 0.035,
            "온톨로지 SignalNode", ha="center", va="center",
            color=C_TEXT, fontsize=11, fontweight="bold")
    ax.text(sig_x + sig_w / 2, sig_y + sig_h / 2,
            "9 룰", ha="center", va="center",
            color=C_ACCENT_2, fontsize=10, fontfamily="monospace")
    ax.text(sig_x + sig_w / 2, sig_y + 0.025,
            "rule · signal_type · confidence",
            ha="center", va="center", color=C_MUTED, fontsize=8.5)

    # 3 boxes — Left/Center/Right
    box_y = 0.28
    box_w = 0.30
    box_h = 0.32

    # LONG STRADDLE (좌측, violet)
    str_x = 0.04
    p = FancyBboxPatch((str_x, box_y), box_w, box_h,
                       boxstyle="round,pad=0.02,rounding_size=0.04",
                       linewidth=1.8, edgecolor="#8b5cf6", facecolor=C_PANEL)
    ax.add_patch(p)
    ax.text(str_x + box_w / 2, box_y + box_h - 0.035, "LONG STRADDLE",
            ha="center", color="#8b5cf6", fontsize=13, fontweight="bold")
    ax.text(str_x + box_w / 2, box_y + box_h - 0.065,
            "변동 크기 매수 (위기 룰 + event_vol)",
            ha="center", color=C_TEXT, fontsize=9.5)
    ax.text(str_x + 0.02, box_y + box_h - 0.105,
            "베헤이클: 롱 ATM 콜+풋", color=C_MUTED, fontsize=9)
    ax.text(str_x + 0.02, box_y + box_h - 0.135,
            "예산: 3% × scale × mult", color=C_MUTED, fontsize=9)
    ax.text(str_x + 0.02, box_y + box_h - 0.165,
            "트레일 arm25%/반납40% / DTE≤10", color=C_MUTED, fontsize=9)
    ax.text(str_x + 0.02, box_y + box_h - 0.200,
            "트리거: natural_hedge, crash_vulnerable,", color="#a78bfa", fontsize=8.5)
    ax.text(str_x + 0.02, box_y + box_h - 0.220,
            "co_crash_cluster, fat_tail_alert (high_vix)", color="#a78bfa", fontsize=8.5)
    ax.text(str_x + 0.02, box_y + box_h - 0.245,
            "+ event_vol ★R9 (전 레짐):", color="#c4b5fd", fontsize=8.5, fontweight="bold")
    ax.text(str_x + 0.02, box_y + box_h - 0.265,
            "  realized z≥2.5 & VRP_iv≤0", color="#a78bfa", fontsize=8.5)
    ax.text(str_x + box_w / 2, box_y + 0.020,
            "P&L = 변동 일어났나?", color=C_ACCENT_3,
            fontsize=9.5, fontweight="bold", ha="center")

    # DIRECTIONAL (중앙, cyan)
    dir_x = 0.35
    p = FancyBboxPatch((dir_x, box_y), box_w, box_h,
                       boxstyle="round,pad=0.02,rounding_size=0.04",
                       linewidth=1.8, edgecolor=C_ACCENT_2, facecolor=C_PANEL)
    ax.add_patch(p)
    ax.text(dir_x + box_w / 2, box_y + box_h - 0.035, "DIRECTIONAL",
            ha="center", color=C_ACCENT_2, fontsize=13, fontweight="bold")
    ax.text(dir_x + box_w / 2, box_y + box_h - 0.065,
            "방향 베팅 (rate beta)",
            ha="center", color=C_TEXT, fontsize=9.5)
    ax.text(dir_x + 0.02, box_y + box_h - 0.105,
            "베헤이클: 섹터 현물 + β·SPY 헤지", color=C_MUTED, fontsize=9)
    ax.text(dir_x + 0.02, box_y + box_h - 0.135,
            "예산: 5% × scale × mult", color=C_MUTED, fontsize=9)
    ax.text(dir_x + 0.02, box_y + box_h - 0.165,
            "페어 P&L: TP+10%/SL-7%/21일", color=C_MUTED, fontsize=9)
    ax.text(dir_x + 0.02, box_y + box_h - 0.205,
            "트리거: rate_beneficiary,", color="#7dd3fc", fontsize=8.5)
    ax.text(dir_x + 0.02, box_y + box_h - 0.225,
            "rate_victim,", color="#7dd3fc", fontsize=8.5)
    ax.text(dir_x + 0.02, box_y + box_h - 0.245,
            "normal_var_inadequate", color="#7dd3fc", fontsize=8.5)
    ax.text(dir_x + 0.02, box_y + box_h - 0.265,
            "(vol_overpriced 제외)", color=C_MUTED, fontsize=8.5)
    ax.text(dir_x + box_w / 2, box_y + 0.020,
            "P&L = 방향 맞았나?", color=C_ACCENT_3,
            fontsize=9.5, fontweight="bold", ha="center")

    # SHORT STRADDLE (우측, emerald NEW)
    sh_x = 0.66
    p = FancyBboxPatch((sh_x, box_y), box_w, box_h,
                       boxstyle="round,pad=0.02,rounding_size=0.04",
                       linewidth=2.2, edgecolor=C_ACCENT_3, facecolor=C_PANEL)
    ax.add_patch(p)
    ax.text(sh_x + box_w / 2, box_y + box_h - 0.035, "SHORT STRADDLE ⭐",
            ha="center", color=C_ACCENT_3, fontsize=13, fontweight="bold")
    ax.text(sh_x + box_w / 2, box_y + box_h - 0.065,
            "변동 크기 매도 (그린라이트)",
            ha="center", color=C_TEXT, fontsize=9.5)
    ax.text(sh_x + 0.02, box_y + box_h - 0.105,
            "베헤이클: 숏 ATM 콜+풋", color=C_MUTED, fontsize=9)
    ax.text(sh_x + 0.02, box_y + box_h - 0.135,
            "예산: 1% × scale (보수)", color=C_MUTED, fontsize=9)
    ax.text(sh_x + 0.02, box_y + box_h - 0.165,
            "TP -50% / SL +100% / DTE≤10", color=C_MUTED, fontsize=9)
    ax.text(sh_x + 0.02, box_y + box_h - 0.205,
            "트리거: vol_overpriced", color="#6ee7b7", fontsize=8.5)
    ax.text(sh_x + 0.02, box_y + box_h - 0.225,
            "    ∩", color=C_ACCENT_3, fontsize=10, fontweight="bold")
    ax.text(sh_x + 0.02, box_y + box_h - 0.245,
            "thin_tail_greenlight", color="#6ee7b7", fontsize=8.5)
    ax.text(sh_x + 0.02, box_y + box_h - 0.265,
            "(ξ<0.10 & λ_L<0.20 & CF<1.5)", color=C_MUTED, fontsize=8)
    ax.text(sh_x + box_w / 2, box_y + 0.020,
            "P&L = IV crush + 무변동", color=C_ACCENT_3,
            fontsize=9.5, fontweight="bold", ha="center")

    # Arrows from signal to all 3 boxes
    _arrow(ax, sig_x + 0.03, sig_y, str_x + box_w * 0.7, box_y + box_h,
           color="#8b5cf6", lw=2.0)
    _arrow(ax, sig_x + sig_w / 2, sig_y, dir_x + box_w / 2, box_y + box_h,
           color=C_ACCENT_2, lw=2.0)
    _arrow(ax, sig_x + sig_w - 0.03, sig_y, sh_x + box_w * 0.3, box_y + box_h,
           color=C_ACCENT_3, lw=2.2)

    # Bottom — 시스템 테제 코히어런스 요약
    ax.text(0.04, 0.215, "시스템 테제 코히어런스 (2026-06-16 현황)",
            color=C_ACCENT_4, fontsize=12, fontweight="bold")
    ax.text(0.04, 0.175,
            "현재 mid_vix:  LONG STRADDLE 1건 (event_vol XLB)  ·  DIRECTIONAL 3건 (rate_victim XLC/XLRE/XLY 숏)  ·  "
            "SHORT STRADDLE 0건",
            color=C_TEXT, fontsize=10, fontweight="bold")
    ax.text(0.04, 0.140,
            "→ event_vol 이 VIX 평온(17.7)에도 실현변동 급등(XLB z=3.0)+싼IV 포착 = VIX 비대칭 보완  ·  "
            "rate_beneficiary KILLED 청산(G1)  ·  그린라이트 死코드",
            color=C_MUTED, fontsize=9.5)
    ax.text(0.04, 0.105,
            "→ 평상시 검증된 directional + 실현변동 이벤트엔 event_vol LONG, 위기엔 위기룰 LONG, 진짜 안전 시 SHORT",
            color=C_ACCENT_3, fontsize=10, fontweight="bold")
    ax.text(0.04, 0.060,
            "feedback (rule × strategy) 키 학습 — 표본 게이트는 cohort(유효표본) 기준: 같은 날 동시진입 = 1 증거 "
            "(MIN=5, clip 0.7~1.3, stop-rule n_eff≥10 & avg<-5% → mult=0)",
            color=C_MUTED, fontsize=9)
    ax.text(0.04, 0.025,
            "수명주기 가드 (06-12): 룰 은퇴 시 보유 포지션 즉시 청산 · 1계약 비용 > 예산×2 면 진입 스킵 · "
            "레짐 전환은 2사이클 연속 확인 (히스테리시스)",
            color=C_MUTED, fontsize=9)

    _footer(fig, n, total)
    pdf.savefig(fig, facecolor=C_BG); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Page 11 — Kinetic feedback loop
# ──────────────────────────────────────────────────────────────

def page_kinetic_loop(pdf, n, total):
    fig = _new_page(figsize=(12.0, 9.0))
    _title(fig, "Kinetic Layer — 시그널 → 주문 → 저널 → 학습 루프",
           "Palantir-style 운영 평면. 분석에서 실행으로 — 그리고 실현 P&L 이 다시 사이징을 보정")

    ax = _blank_axes(fig, (0.04, 0.04, 0.92, 0.82))

    # 5 nodes in circular layout
    nodes = [
        ("trigger.py\nSignal → 듀얼 후보",            0.50, 0.85, C_ACCENT_1),
        ("kinetic_executor.py\nAlpaca 진입/청산",      0.85, 0.55, C_ACCENT_3),
        ("trade_journal.py\nappend-only JSONL",        0.70, 0.18, "#fbbf24"),
        ("attribution.py\n(rule × strategy) 귀인",     0.30, 0.18, C_ACCENT_2),
        ("feedback.py\nrule_performance.json",          0.15, 0.55, C_ACCENT_3),
    ]
    w, h = 0.21, 0.10
    pos = []
    for txt, x, y, color in nodes:
        p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                           boxstyle="round,pad=0.02,rounding_size=0.04",
                           linewidth=1.8, edgecolor=color, facecolor=C_PANEL)
        ax.add_patch(p)
        # txt 첫 줄과 둘째 줄 분리
        first, *rest = txt.split("\n")
        ax.text(x, y + 0.018, first, ha="center", va="center",
                color=color, fontsize=11, fontweight="bold",
                fontfamily="monospace")
        if rest:
            ax.text(x, y - 0.022, rest[0], ha="center", va="center",
                    color=C_TEXT, fontsize=9.5)
        pos.append((x, y))

    # Arrows in clockwise order. 라벨은 화살표 중점에서 노멀 방향으로 살짝 밀어 표기.
    labels = [
        "ontology_signals.json",
        "주문/체결",
        "거래 이벤트",
        "완결 거래",
        "사이징 승수\n(다음 진입 반영)",
    ]
    for i in range(5):
        (x1, y1) = pos[i]
        (x2, y2) = pos[(i + 1) % 5]
        dx, dy = x2 - x1, y2 - y1
        L = np.hypot(dx, dy)
        ox, oy = dx / L * 0.13, dy / L * 0.06
        _arrow(ax, x1 + ox, y1 + oy, x2 - ox, y2 - oy,
               color=C_ACCENT_4, lw=2.0)
        # 라벨을 화살표 노멀(밖) 방향으로 오프셋해 겹침 방지
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        cx, cy = 0.50, 0.50   # 다이어그램 중심
        nx, ny = mx - cx, my - cy
        nL = max(np.hypot(nx, ny), 1e-6)
        off = 0.055
        lx, ly = mx + nx / nL * off, my + ny / nL * off
        ax.text(lx, ly, labels[i], ha="center", va="center",
                color=C_ACCENT_4, fontsize=9, linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=C_BG,
                          edgecolor="none", alpha=0.85))

    # Center text
    ax.text(0.50, 0.50, "루프 폐쇄\n(실현 P&L → 학습)",
            ha="center", va="center", color=C_TEXT, fontsize=13,
            fontweight="bold", style="italic")

    # Bottom: key constraints
    ax.text(0.04, 0.065,
            "제약: 옵션 마켓 주문은 RTH(09:30–16:00 ET) 만 가능. 휴장 시 자동 가드. "
            ".env 자격증명 (코드 하드코딩 금지). 피드백은 사이징 승수만 자동 — 임계값 T 와 청산 룰은 수동.",
            color=C_MUTED, fontsize=9.5)
    ax.text(0.04, 0.030,
            "수명주기 (2026-06-12): ① 룰은퇴 청산 — 룰셋 제거/stop-rule 포지션은 TP/DTE 대기 없이 즉시 회수  "
            "② 예산 가드 — 최소 1계약이 예산×2 초과 시 진입 스킵 (사이징 역전 방지)  "
            "③ cohort 유효표본 — 동시진입 상관 거래를 1 증거로 집계",
            color=C_ACCENT_3, fontsize=9)

    _footer(fig, n, total)
    pdf.savefig(fig, facecolor=C_BG); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Page — Backtest results (NEW 2026-06-10)
# ──────────────────────────────────────────────────────────────

def page_backtest(pdf, n, total):
    """backtest_results.json 라이브 로드 — 하드코딩 숫자가 페이지끼리 어긋나던
    문제(라운드 7 비평 4번)의 영구 해결: 이 페이지의 모든 수치는 마지막 실행값."""
    bt = {}
    try:
        bt = json.loads((OUTPUT_DIR / "backtest_results.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    period = bt.get("period", {})
    trail = bt.get("trail", {})
    tp25 = bt.get("tp25", {})
    base = bt.get("baseline_spy", {})
    scan = bt.get("trail_sensitivity", [])

    fig = _new_page(figsize=(12.0, 9.0))
    _title(fig, f"백테스트 — {period.get('start','?')} ~ {period.get('end','?')} "
                f"({period.get('n_days','?')}일 · {trail.get('n_closed','?')} closed, 9룰·FDR 게이트)",
           f"거래 수 이력: 라운드6 194 closed → 룰 가지치기(11→8)+FDR 게이트 → R9 event_vol(+1룰)로 "
           f"{trail.get('n_closed','?')} closed(straddle 15 + directional 28, event_vol n=2 포함) · "
           "BS+GARCH IV · 슬리피지 5% · ⚠ 라운드6-7 in-sample 재구성 ¤ — 전향 검증 아님(생존편향 상향)")

    ax = _blank_axes(fig, (0.04, 0.06, 0.92, 0.82))

    # 청산 구조 대조 (동일 시그널 스트림)
    ax.text(0.02, 0.97, "STRADDLE 청산 구조 대조 (동일 시그널, in-sample)",
            color=C_ACCENT_4, fontsize=12, fontweight="bold")
    s_tp = (tp25.get("by_strategy") or {}).get("straddle", {})
    s_tr = (trail.get("by_strategy") or {}).get("straddle", {})
    d_tr = (trail.get("by_strategy") or {}).get("directional", {})
    y = 0.93
    for tag, v, c in [("tp25 (구)", s_tp, C_MUTED), ("trail (현행)", s_tr, C_ACCENT_3),
                      ("directional", d_tr, C_ACCENT_2)]:
        if not v:
            continue
        ax.text(0.02, y, f"{tag:14s} n={v.get('trades',0):>3}  "
                f"win={v.get('win_rate',0)*100:>3.0f}%  "
                f"avg={v.get('avg_pnl_pct',0)*100:+5.1f}%  "
                f"total=${v.get('total_pnl',0):+8,.0f}",
                color=c, fontsize=10.5, fontfamily="monospace", fontweight="bold")
        y -= 0.035

    # 시스템 합산(straddle+directional) — W3 의 +$1,252. ★ in-sample 재구성 라벨 필수
    sys_tot = (s_tr.get("total_pnl", 0) or 0) + (d_tr.get("total_pnl", 0) or 0)
    ax.text(0.02, y, f"시스템 합산  total={sys_tot:+,.0f} USD   "
            f"(라운드6-7 in-sample 재구성 ¤ — 전향 검증 아님)",
            color=C_ACCENT_5, fontsize=10, fontweight="bold")
    y -= 0.045

    # 베이스라인 — 라운드 6/7 비평의 결정적 비교
    ax.text(0.02, y - 0.01, "베이스라인 대조군 (시그널 없음): " + base.get("rule", "미실행"),
            color=C_ACCENT_5, fontsize=11, fontweight="bold")
    y -= 0.045
    if base:
        beat = s_tr.get("total_pnl", 0) > base.get("total_pnl", 0)
        n_base = base.get("trades", 0)
        # 에피스테믹 대칭(R8 검토 1): n<10 이면 시스템 패배도, 대조군 숫자(+3,409)도 둘 다
        # 주장 불가 — n=2 caveat 를 한쪽(시스템 열위)에만 붙이면 비대칭. 방향만 시사.
        if n_base < 10:
            verdict = (f"→ n={n_base} 라 양쪽 다 판정 불가(대조군 숫자도 무의미), "
                       f"단 방향은 시스템 열위 시사")
        else:
            verdict = "→ 시스템이 " + ("이김" if beat else "짐 — 6레이어의 한계 기여 미증명")
        ax.text(0.02, y, f"{'baseline':14s} n={n_base:>3}  "
                f"win={base.get('win_rate',0)*100:>3.0f}%  "
                f"avg={base.get('avg_pnl_pct',0)*100:+5.1f}%  "
                f"total={base.get('total_pnl',0):+,.0f} USD   " + verdict,
                color=C_ACCENT_5,
                fontsize=10.5, fontfamily="monospace", fontweight="bold")
    y -= 0.05

    # (rule × strategy) — trail 기준, 에피스테믹 라벨은 n 으로 기계 판정
    ax.text(0.02, y, "(rule × strategy) — trail 기준 · n<10 은 전부 '후보/미검증' (이항 검정 불가)",
            color=C_ACCENT_4, fontsize=11, fontweight="bold")
    y -= 0.035
    rs = trail.get("by_rule_strategy", {})
    for key, v in sorted(rs.items(), key=lambda kv: -kv[1].get("total_pnl", 0)):
        n_t = v.get("trades", 0)
        label = "후보 (n<10, 주장 불가)" if n_t < 10 else "표본 누적 중"
        pnl = v.get("total_pnl", 0)
        c = C_ACCENT_3 if pnl > 0 else C_ACCENT_5
        ax.axhline(y + 0.013, xmin=0.02, xmax=0.98, color=C_LIGHT, lw=0.4, alpha=0.3)
        ax.text(0.02, y, key, color=C_TEXT, fontsize=9, fontfamily="monospace")
        ax.text(0.42, y, f"n={n_t:>3}  win={v.get('win_rate',0)*100:>3.0f}%  "
                f"avg={v.get('avg_pnl_pct',0)*100:+5.1f}%  ${pnl:+8,.0f}",
                color=c, fontsize=9, fontfamily="monospace")
        ax.text(0.80, y, label, color=C_MUTED, fontsize=8.5)
        y -= 0.033

    # 숫자 정합 (P2-3): straddle total = co_crash + fat_tail 항등식 + 레거시 reconcile
    cc = (trail.get("by_rule_strategy") or {}).get("co_crash_cluster|straddle", {}).get("total_pnl", 0)
    ft = (trail.get("by_rule_strategy") or {}).get("fat_tail_alert|straddle", {}).get("total_pnl", 0)
    y -= 0.015
    ax.text(0.02, y,
            f"숫자 정합(P2-3, USD): trail straddle {s_tr.get('total_pnl',0):+,.0f} "
            f"= co_crash {cc:+,.0f} + fat_tail {ft:+,.0f} (항등식 성립). "
            f"감사 레거시 +726/+1,210/+1,639 는 라운드6-7 stale 스냅샷 — 당시 total≠룰합 "
            f"(+1,639 vs +1,210) 불일치는 현 데이터에서 해소됨.",
            color=C_MUTED, fontsize=8.5, style="italic")
    y -= 0.03

    # 트레일 민감도 격자 (운 vs 강건)
    y -= 0.02
    ax.text(0.02, y, "트레일 파라미터 민감도 (straddle total USD — 인접 격자에서 부호 유지 = 강건)",
            color=C_ACCENT_4, fontsize=11, fontweight="bold")
    y -= 0.035
    if scan:
        ax.text(0.02, y, f"{'arm/gb':>7} {'30%':>10} {'40%':>10} {'50%':>10}",
                color=C_MUTED, fontsize=9.5, fontfamily="monospace")
        y -= 0.03
        for arm in (0.20, 0.25, 0.30):
            cells = []
            for gb in (0.30, 0.40, 0.50):
                cell = next((c for c in scan
                             if abs(c["arm"] - arm) < 1e-9 and abs(c["giveback"] - gb) < 1e-9), None)
                cells.append(f"{cell['total_pnl']:+10,.0f}" if cell else f"{'-':>10}")
            mark = " ← 현행" if abs(arm - 0.25) < 1e-9 else ""
            ax.text(0.02, y, f"{arm*100:>6.0f}% {cells[0]} {cells[1]} {cells[2]}{mark}",
                    color=C_TEXT, fontsize=9.5, fontfamily="monospace")
            y -= 0.03
        # 튜닝 여부 명기 (P2-3): 현행 25/40 은 격자 최대 아님 → 보수적 선택
        vals = [c.get("total_pnl", 0) for c in scan]
        cur = next((c for c in scan if abs(c["arm"]-0.25) < 1e-9 and abs(c["giveback"]-0.40) < 1e-9), None)
        best = max(scan, key=lambda c: c.get("total_pnl", 0)) if scan else None
        if cur and best:
            ax.text(0.02, y - 0.005,
                    f"튜닝 여부(USD): 스캔 실행됨(파라미터 공간 in-sample 관측). 현행 25/40="
                    f"{cur['total_pnl']:+,.0f} 은 격자 최대 아님(최대 "
                    f"{best['arm']*100:.0f}/{best['giveback']*100:.0f}="
                    f"{best['total_pnl']:+,.0f}) = 보수적 선택. 9격자 전부 양(+), "
                    f"부호 강건 — 단 n=13 소표본.",
                    color=C_MUTED, fontsize=8.5, style="italic")
            y -= 0.035
    else:
        ax.text(0.02, y, "(스캔 미실행)", color=C_MUTED, fontsize=9)
        y -= 0.03

    ax.text(0.02, 0.025,
            "⚠ 이 페이지의 모든 수치는 같은 6개월 표본 위의 in-sample 산출 — 룰 가지치기(11→8)가 "
            "이 표본에서 이뤄졌으므로 생존편향 상향. 검증 가능한 숫자는 config freeze 이후의 페이퍼 표본뿐.",
            color=C_ACCENT_4, fontsize=9, style="italic")

    _footer(fig, n, total)
    pdf.savefig(fig, facecolor=C_BG); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Page 12 — Known limitations (외부 비평 반영)
# ──────────────────────────────────────────────────────────────

def page_round78(pdf, n, total):
    fig = _new_page(figsize=(12.0, 9.0))
    _title(fig, "라운드 7-8 결론 — 감지는 검증, 수익화는 스프레드",
           "외부 감사 P0~W3 반영. 모든 손익은 in-sample ¤ — 전향 페이퍼 표본만 진짜 검증.")

    ax = _blank_axes(fig, (0.04, 0.06, 0.92, 0.82))

    items = [
        ("1. 감지 레이어는 가치 있음 (P1-1)",
         "하락 위기 홈그라운드 시험: 게이트가 B&H 대비 낙폭 대폭 축소 = 브레이크로 작동.\n"
         "GFC MDD -19% vs B&H -74% · COVID -10% vs -60% · 긴축 -12% vs -33%. 게이트 무가치 아님.",
         C_ACCENT_3),

        ("2. 수익화 = β헤지 스프레드 (W2-A · IV 가정 0)",
         "게이트 OPEN 시 액션 비교: ③ β헤지 스프레드가 현행 on/off 를 4창 전부 MDD 우위,\n"
         "한 줄짜리 200DMA 를 3/4창 우위(호르무즈 -5%·GFC -7%·긴축 -2%). ② 노출가변 1.5x 는 재앙(GFC -76%).\n"
         "→ 수익화 매핑은 '미해결'이 아니라 '잘못된 instrument' — 답(스프레드)이 데이터로 나옴.",
         C_ACCENT_3),

        ("3. 스트래들 기각 (W2-B · GARCH-IV 배수 민감도)",
         "ATM 스트래들 GARCH-IV×{1.0,1.5,2.0}: 호르무즈 +95%→-14%(2x 손실), 하락 3창 전 배수 손실.\n"
         "위기 IV > 합성가는 구조적 → 라이브 instrument(스트래들)는 현실 IV(1.5~2배)에서 죽는다.",
         C_ACCENT_5),

        ("4. P0 — rate_* FDR 게이트 강제 (US2Y 뒷문 차단)",
         "rate_beneficiary 가 US2Y |t|>1.96 폴백으로 FDR 우회 발화하던 결함 차단(q<0.10 강제).\n"
         "라이브 directional = rate_victim ALIVE 3종(XLRE/XLY/XLC)만. SSOT 자기정합 + 체커 재발방지.",
         C_ACCENT_3),

        ("5. ML shadow 격리 (W1 · 결정 D안)",
         "ML(LR/RF/GBT·VQC)은 라이브 사이징 경로에 부재 확정 — 평가 전용. 발화만 누적(n_fire=2/30),\n"
         "사이징 미반영. 제품명 'Ontology-Gated Framework (ML in shadow eval)'. n≥30 도달 시 재결정.",
         C_ACCENT_2),

        ("6. 확정 결정 + 라벨 (D1=가 / P2-4=A)",
         "directional → β헤지 스프레드 전환 확정(G4 freeze 시 적용, 126D β). TP/SL=상수배수×진입 σ_resid.\n"
         "전제 정정: 시스템 +1,252 USD 도 라운드6-7 in-sample 재구성 ¤ — '검증된 성과' 아님. freeze 후\n"
         "전향 페이퍼 표본만 유효 OOS. pending_forward_validation 에 미검증 주장 목록 등재(G4).",
         C_ACCENT_1),
    ]

    y = 0.97
    for title, body, color in items:
        p = FancyBboxPatch((0.0, y - 0.022), 0.98, 0.035,
                           boxstyle="round,pad=0.003,rounding_size=0.02",
                           linewidth=1.4, edgecolor=color, facecolor=C_PANEL)
        ax.add_patch(p)
        ax.text(0.01, y - 0.005, title, color=color, fontsize=11,
                fontweight="bold", va="center")
        ax.text(0.02, y - 0.035, body, color=C_TEXT, fontsize=8.5,
                linespacing=1.4, va="top")
        y -= 0.155

    ax.text(0.5, 0.005,
            "★ R9(2026-06-16): event_vol 룰 신설(실현변동성→롱 볼, VIX 비대칭 보완) 후 재freeze. "
            "이제 데이터 대기 — 전향 페이퍼 표본(event_vol 포함 6주장 PENDING)만이 유일한 진실.",
            ha="center", color=C_ACCENT_3, fontsize=9.5, fontweight="bold")

    _footer(fig, n, total)
    pdf.savefig(fig, facecolor=C_BG); plt.close(fig)


def page_limitations(pdf, n, total):
    fig = _new_page(figsize=(12.0, 9.0))
    _title(fig, "알려진 구조적 한계 (2026-06-12 라운드 6 반영)",
           "라운드 4~6 비평 누적 반영. 신규 해결 5 / 남은 구조적 한계 2.")

    ax = _blank_axes(fig, (0.04, 0.06, 0.92, 0.82))

    items = [
        ("0. ✓ 누적 해결 (라운드 4~5 + 수명주기 3종, 06-08~12)",
         "VRP_true = ATM_IV − RV (Alpaca). 정합성 의심 3룰 제거 (11→8). 룰은퇴 청산 / 예산 가드 /\n"
         "레짐 히스테리시스. cohort 유효표본 (동시진입 = 1 증거).",
         C_ACCENT_3),

        ("1. ✓ TP +25% 꼬리 절단 폐지 → 트레일링 (라운드 6 비평 1번)",
         "fat-tail 탐지 시스템이 컨벡시티 수익을 캡에서 자르던 자기모순 해소. arm 25% / 피크 40% 반납.\n"
         "동일 시그널 백테스트: tp25 → trail 로 straddle 개선(현 수치·정합 항등식은 백테스트 페이지 참조). "
         "⚠ 같은 표본 in-sample 비교(레거시 +726/+1,639 는 stale 스냅샷).",
         C_ACCENT_3),

        ("2. ✓ 검증-실행 불일치 해소 → β·SPY 헤지 페어 (라운드 6 비평 2번)",
         "partial 회귀가 검증한 건 시장-대비 민감도인데 실행은 아웃라이트 숏이었음 (P&L = -β·R_SPY 지배).\n"
         "진입 시 SPY 반대 레그 (단순 126D β — partial beta 는 VIX 가 흡수해 과소헤지), 페어 P&L 로 채점.",
         C_ACCENT_3),

        ("3. ✓ 추정 노이즈 보수화 + 공선성/FDR (라운드 6 비평 4·7번)",
         "ξ·λ_L·CF 블록 부트스트랩 CI — thin_tail_greenlight(무한손실 게이트)는 95% 상한으로만 발화.\n"
         "금리 직교화(레벨+2s10s)+VIF: XLE 금리수혜 kill 은 교란 확정, XLV/XLP rate_victim 은 직교화 후\n"
         "t 유의성 소멸로 사망(KILLED, ctrl t=-0.1/-1.6 — p.9 표와 동일 사유). verdict 는 BH FDR q<0.10.",
         C_ACCENT_3),

        ("4. (남은 한계) 생존편향 + 열린 루프 + 노출 캡 부재  [R9 일부 완화]",
         "11→8 가지치기 자체가 같은 백테스트의 in-sample 선택 — 생존 룰 수치는 상향 편향 (라운드 6 비평 5번).\n"
         "수치 검증은 지금부터의 페이퍼 표본(진짜 OOS)으로만. high_vix 룰 표본 누적은 수년 — R9 event_vol 이\n"
         "레짐 독립 발화라 평상시에도 표본 누적(열린 루프 완화). 단 룰 패밀리별 합산 노출 캡은 여전히 미구현.",
         C_ACCENT_1),

        ("5. (남은 한계) look-ahead, n=8 통계, 실체결 낙관 + 베이스라인 패배",
         "tail/co_crash 통계 시점-aware 재적합 미구현 → co_crash 75%/n=8 은 이항 p≈0.15 — 알파 '후보' 일 뿐\n"
         "주장 불가 (라운드 6 비평 3번). 섹터 ETF 옵션 실스프레드는 5% 가정보다 나쁜 날 많음 (페이퍼 낙관).\n"
         "VIX>25 SPY 대조군(R8/W3): 시스템 +1,252 < 대조군 +3,409 (USD) — 시스템이 졌다(n=2 caveat·방향 분명).",
         "#a78bfa"),
    ]

    y = 0.97
    for title, body, color in items:
        # 제목 박스
        p = FancyBboxPatch((0.0, y - 0.022), 0.98, 0.035,
                           boxstyle="round,pad=0.003,rounding_size=0.02",
                           linewidth=1.4, edgecolor=color, facecolor=C_PANEL)
        ax.add_patch(p)
        ax.text(0.01, y - 0.005, title, color=color, fontsize=11,
                fontweight="bold", va="center")
        # 본문
        ax.text(0.02, y - 0.035, body, color=C_TEXT, fontsize=8.5,
                linespacing=1.4, va="top")
        y -= 0.155

    ax.text(0.5, 0.022,
            "★ CONFIG FREEZE v1 (2026-06-16, R9 갱신): 9룰(+event_vol) + FDR q<0.10 게이트 + 트레일 25/40 "
            "+ β페어 σ임계(2.0/1.5) + rate 캡 15% + 숏볼 휴면. OOS 전향 시계 = 이 날짜부터 (구조변경 리셋).",
            ha="center", color=C_ACCENT_3, fontsize=9.5, fontweight="bold")
    ax.text(0.5, 0.004,
            "⚠ freeze v1 은 paper-guard 미구현 상태를 동결한다 (TRADE_BASE=env[ALPACA_ENDPOINT], 무가드). "
            "실자본 계좌 연결일 = freeze v1 파기 + v2 강제 트리거 — 동결된 채 잊히지 않게 못 박음.",
            ha="center", color=C_ACCENT_5, fontsize=9, fontweight="bold")

    _footer(fig, n, total)
    pdf.savefig(fig, facecolor=C_BG); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Page 13 — Glossary
# ──────────────────────────────────────────────────────────────

def page_glossary(pdf, n, total):
    fig = _new_page(figsize=(12.0, 9.0))
    _title(fig, "용어집 / 약어 사전",
           "본 명세서에서 사용되는 핵심 개념")

    ax = _blank_axes(fig, (0.04, 0.06, 0.92, 0.82))

    items = [
        ("GARCH(1,1)",            "Generalized AutoRegressive Conditional Heteroskedasticity. 시점별 변동성을 모델링하는 시계열 모형."),
        ("VRP",                   "Volatility Risk Premium. 내재변동성(IV) - 실현변동성(RV). 양수면 시장이 헤지에 과한 프리미엄 지불."),
        ("IV / RV",               "Implied / Realized Volatility. 옵션 가격이 함의하는 vs 과거 데이터로 측정한 변동성."),
        ("EVT",                   "Extreme Value Theory. 극단치만 별도 모델링하는 통계 이론."),
        ("GPD",                   "Generalized Pareto Distribution. 임계값(POT) 초과분에 적합하는 분포."),
        ("POT",                   "Peaks Over Threshold. 임계값 초과 사건만으로 꼬리 적합."),
        ("ξ (xi)",                "GPD 꼬리 지수. >0 → fat tail. >0.5 → 분산 무한. >1.0 → 평균 무한."),
        ("VaR / ES (99%)",        "Value at Risk / Expected Shortfall — 1% 확률의 손실 한계 / 그 너머 평균 손실."),
        ("CF VaR",                "Cornish-Fisher VaR. 왜도·첨도로 정규 quantile 을 보정한 VaR."),
        ("Copula",                "주변분포와 의존구조를 분리해 모델링하는 함수족. 본 시스템은 비모수 λ 직접 계산."),
        ("λ_L / λ_U",             "Lower / Upper Tail Dependence. 동시 폭락/급등 확률. 0=무관, 1=항상 함께."),
        ("Cholesky 분해",         "Σ = L L^T. 대칭 양정부호 행렬을 하삼각 인자로 분해. SVAR-style 재귀 분산 분해의 기반."),
        ("var_attr[i,j]",         "Cholesky 분해에서 j 의 직교 충격이 i 분산에 기여한 비율 (행합=1)."),
        ("NORTA",                 "NORmal-To-Anything. 가우시안 공간에서 의존성을 주입한 뒤 마진을 자유롭게 변환하는 시뮬 기법."),
        ("PCMCI",                 "Peter-Clark Momentary Conditional Independence. Granger 의 허위 인과 문제를 해결한 인과 발견."),
        ("structural / emerging", "PCMCI 링크 안정성. 전체기간 + 현재레짐 양쪽 유의 / 현재레짐만 유의."),
        ("SignalNode",            "온톨로지가 발화한 시그널 객체. (sector, signal_type, confidence, reasoning, rule_name)."),
        ("signal_type",           "OVERWEIGHT (+1) · UNDERWEIGHT (-1) · HEDGE (-1) · MONITOR (0)."),
        ("듀얼 전략",             "같은 SignalNode 를 STRADDLE(크기) + DIRECTIONAL(방향) 두 가지로 동시 베팅."),
        ("STRADDLE",              "롱 ATM 콜 + 풋 동시 보유. 방향 무관, 변동 크기에서 수익."),
        ("trade_id",              "거래 진입↔청산을 연결하는 UUID 6자리. 매칭 키 = (ticker, strategy)."),
        ("attribution",           "거래 저널을 (rule × strategy) 단위로 집계해 룰별 실현 성과 산출."),
        ("rule_performance.json", "feedback 산출물. 룰|전략 별 사이징 승수 (0.5 ~ 1.5 clip)."),
        ("RTH",                   "Regular Trading Hours. 09:30–16:00 ET. 옵션 주문은 이 시간에만 가능."),
        ("OCC 심볼",              "옵션 체인 표준 식별자. ROOT + YYMMDD + C/P + strike×1000 (8자리)."),
        ("base rate",             "시그널 없이도 시장 구조만으로 발생할 확률. NORTA Joint DD 가 이를 산출."),
        ("delta_ctrl (partial)",  "전 매크로(OIL 포함)+SPY 동시 통제 후의 한계 민감도. rate_* 룰의 발화 기준 (06-12~)."),
        ("cohort (n_eff)",        "같은 (rule|strategy|진입일) 동시진입 묶음 = 유효표본 1개. feedback 게이트의 표본 단위."),
        ("레짐 히스테리시스",      "새 VIX 레짐이 연속 2사이클 관측돼야 전환. regime_state.json 에 confirmed/pending 유지."),
        ("룰은퇴 청산",            "룰셋 제거 또는 stop-rule(mult=0) 된 룰의 보유 포지션을 즉시 회수하는 청산 사유."),
        ("트레일링 청산",          "arm(+25%) 후 피크 이익의 40% 반납 시 청산. 고정 TP 의 우측 꼬리 절단 해소 (06-12)."),
        ("BH FDR q값",            "Benjamini-Hochberg 다중비교 보정. 55개 민감도 가설의 verdict 는 q<0.10 기준."),
        ("β 헤지 페어",            "섹터 현물 + 단순 126D β 만큼 SPY 반대 레그. 검증된 시장-대비 효과를 그대로 거래."),
        ("event_vol (R9)",        "실현변동성 z≥2.5 & VRP_iv≤0 → 롱 스트래들. VIX 가 놓치는 실현 변동을 싼 IV 일 때 포착."),
        ("rv_zscore",             "실현변동성(10일 RV) 의 1년 기준선 대비 z-score. VIX 와 독립. event_vol 입력."),
    ]

    # 2-column layout — 각 컬럼을 별도 axes 로 두어 axes 폭 안에서 줄바꿈 처리
    half = (len(items) + 1) // 2
    for col, group in enumerate((items[:half], items[half:])):
        # figure 좌표: 좌 0.04~0.49 (폭 0.45), 우 0.52~0.97 (폭 0.45)
        cx = 0.04 + col * 0.48
        cax = fig.add_axes((cx, 0.06, 0.45, 0.78))
        cax.set_xlim(0, 1); cax.set_ylim(0, 1)
        cax.set_facecolor(C_BG)
        for s in cax.spines.values(): s.set_visible(False)
        cax.set_xticks([]); cax.set_yticks([])

        y = 0.98
        for k, v in group:
            cax.text(0.0, y, k, color=C_ACCENT_2, fontsize=9.5,
                     fontweight="bold", fontfamily="monospace", va="top")
            # 정의: axes 폭을 넘으면 줄바꿈을 발생시키기 위해 wrap 한도 직접 지정
            # 길이 60자 넘으면 적절히 짧게 표기
            text_v = v if len(v) <= 80 else v[:80].rsplit(' ', 1)[0] + " ..."
            cax.text(0.0, y - 0.026, text_v, color=C_TEXT, fontsize=8.5,
                     va="top", wrap=True)
            y -= 0.0545

    _footer(fig, n, total)
    pdf.savefig(fig, facecolor=C_BG); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("[overview-pdf] ontology_signals.json 로드...")
    sig_path = OUTPUT_DIR / "ontology_signals.json"
    if not sig_path.exists():
        print(f"  [WARN] {sig_path} 없음 — trigger.py 먼저 실행 필요")
        live = {"variance_decomposition": {"by_sector": []},
                "joint_simulation": None}
    else:
        live = json.loads(sig_path.read_text(encoding="utf-8"))

    print(f"[overview-pdf] PDF 빌드 → {PDF_OUT}")
    total = 16
    with PdfPages(PDF_OUT) as pdf:
        page_cover(pdf, total)
        page_flow(pdf, 2, total)
        page_garch(pdf, 3, total)
        page_evt(pdf, 4, total)
        page_copula(pdf, 5, total)
        page_cholesky(pdf, 6, total, live)
        page_pcmci(pdf, 7, total)
        page_rules(pdf, 8, total)
        page_sensitivity(pdf, 9, total, live)    # NEW 2026-06-12
        page_joint(pdf, 10, total, live)
        page_dual(pdf, 11, total)
        page_kinetic_loop(pdf, 12, total)
        page_backtest(pdf, 13, total)            # NEW 2026-06-10
        page_round78(pdf, 14, total)             # NEW 2026-06-14 (라운드 7-8 결론)
        page_limitations(pdf, 15, total)
        page_glossary(pdf, 16, total)
        d = pdf.infodict()
        d["Title"]    = "Macro Research System Overview"
        d["Author"]   = "wjdrj"
        d["Subject"]  = "6-Layer 온톨로지 + 트리플 전략 + Kinetic Layer + 백테스트 + 민감도 감사"
        d["Keywords"] = "GARCH EVT Copula Cholesky Shapley PCMCI NORTA Straddle Alpaca Backtest Partial-Delta Cohort"

    print(f"[overview-pdf] 완료. 크기={PDF_OUT.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()

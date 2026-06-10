"""
build_overview_pdf.py — 매크로 리서치 시스템 핵심 아이디어 흐름도 PDF 생성.

페이지 구성
─────────
1. 표지 — 비전 + 핵심 질문
2. 전체 흐름도 — 6-Layer + Kinetic Layer
3. Layer 1: GARCH(1,1) — 변동성과 VRP
4. Layer 2: EVT/GPD — Fat-tail
5. Layer 3: Copula — 꼬리 의존성
6. Layer 4: Cholesky 분산 분해 — 실제 XLK 분해 그래프
7. PCMCI 인과 체인
8. Layer 5: Ontology Gate — 11 룰
9. NORTA Joint Drawdown — 시장 base rate (실제 시뮬)
10. 듀얼 전략 — 같은 시그널 두 베팅
11. Kinetic Layer 피드백 루프
12. 용어집

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
    fig.text(0.04, 0.03, "Macro Research System  ·  2026-06-08",
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
        ("① 감지", "6-Layer 온톨로지\n(GARCH→EVT→Copula→Cholesky→Gate→ML)",
         C_ACCENT_1),
        ("② 실행", "트리플 전략 키네틱 레이어\n(롱 스트래들 / 현물 / 숏 스트래들)",
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

    ax.text(0.5, 0.10, "2026-06-08 · 듀얼 전략 + Cholesky 분산 분해 도입 버전",
            ha="center", color=C_MUTED, fontsize=10)

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
        ("Layer 5", "Ontology Gate",              "11 룰 → SignalNode",        C_ACCENT_1),
        ("Layer 6", "ML (XGBoost + VQC)",         "21일 분위 예측",            "#a78bfa"),
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
              "→ 룰 causal_chain_monitor : 섹터가 노출된 매크로의 업스트림 체인 감지",
              ha="center", color=C_ACCENT_1, fontsize=10, fontweight="bold")
    ax_g.text(0.5, 0.005, "stru = structural, emer = emerging",
              ha="center", color=C_MUTED, fontsize=8)

    _footer(fig, n, total)
    pdf.savefig(fig, facecolor=C_BG); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Page 8 — Layer 5 Ontology Gate (rules table)
# ──────────────────────────────────────────────────────────────

def page_rules(pdf, n, total):
    fig = _new_page(figsize=(12.0, 9.0))
    _title(fig, "Layer 5 — Ontology Gate (8 추론 룰, 2026-06-10 정리)",
           "11 → 8 룰. 라운드 5 비평 선반영 — 정합성 의심 3개 제거. thin_tail_greenlight 1개 신설.")

    ax = _blank_axes(fig, (0.04, 0.06, 0.92, 0.80))

    rules = [
        # (rule, condition, signal, meaning, is_new, is_removed)
        ("natural_hedge",          "gamma(VIX) > 0.10",                          "OVERWEIGHT", "VIX 급등 시 볼록 수익", False, False),
        ("crash_vulnerable",       "delta(VIX) < -0.008 & tail_asymm < -0.001",  "UNDERWEIGHT", "하락장 비대칭 손실", False, False),
        ("co_crash_cluster",       "λ_L ≥ 0.40 peer ≥ 3개",                       "HEDGE",       "위기 시 분산 효과 소멸 ✓ +20.4% 백테스트", False, False),
        ("rate_beneficiary",       "delta(US10Y or US2Y) > 0.002",                "OVERWEIGHT", "금리 상승 수혜 ✓ +3.2% 백테스트", False, False),
        ("rate_victim",            "delta(US10Y) < -0.001",                       "UNDERWEIGHT", "장기 듀레이션 피해", False, False),
        ("vol_overpriced",         "VRP_true > 2% (Alpaca IV)",                   "OVERWEIGHT", "IV>RV — directional 제외 + SHORT 페어", False, False),
        ("fat_tail_alert",         "ξ (GPD) > 0.20",                              "HEDGE/MONI",  "정규 VaR 부적합 ✓ +9.4% 백테스트", False, False),
        ("normal_var_inadequate",  "CF VaR / Normal VaR > 2.5",                   "MONITOR",     "정규 VaR 실제의 절반", False, False),
        ("thin_tail_greenlight ⭐",  "ξ<0.10 & λ_L<0.20 & CF<1.5",                  "MONITOR",     "EVT+Copula+CF 트리오 — SHORT STRADDLE 게이트", True, False),
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
        y -= 0.048

    # 레짐별 활성 룰 (8개)
    ax.text(0.02, 0.17, "레짐별 활성 룰 (8개, 2026-06-10)",
            color=C_ACCENT_4, fontsize=12, fontweight="bold")
    regimes = [
        ("low_vix (VIX < 15)",
         "vol_overpriced · rate_beneficiary · thin_tail_greenlight"),
        ("mid_vix (15 ≤ VIX < 25)",
         "rate_* · normal_var_inadequate · vol_overpriced · thin_tail_greenlight"),
        ("high_vix (VIX ≥ 25)",
         "natural_hedge · crash_vulnerable · co_crash_cluster · fat_tail_alert · rate_victim"),
    ]
    y = 0.125
    for k, v in regimes:
        ax.text(0.02, y, k, color=C_ACCENT_2, fontsize=10, fontweight="bold")
        ax.text(0.26, y, v, color=C_TEXT, fontsize=9)
        y -= 0.038

    ax.text(0.02, 0.005,
            "thin_tail_greenlight 는 high_vix 자동 비활성 — 위기 시 숏 볼 위험 회피 (시스템 가설)",
            color=C_MUTED, fontsize=9, style="italic")

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
    _title(fig, "트리플 전략 — 같은 시그널, 세 가지 베팅 (2026-06-10)",
           "LONG STRADDLE (위기 룰) + DIRECTIONAL (rate beta) + SHORT STRADDLE (그린라이트)")

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
            "8 룰", ha="center", va="center",
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
            "변동 크기 매수 (위기 룰)",
            ha="center", color=C_TEXT, fontsize=9.5)
    ax.text(str_x + 0.02, box_y + box_h - 0.105,
            "베헤이클: 롱 ATM 콜+풋", color=C_MUTED, fontsize=9)
    ax.text(str_x + 0.02, box_y + box_h - 0.135,
            "예산: 3% × scale × mult", color=C_MUTED, fontsize=9)
    ax.text(str_x + 0.02, box_y + box_h - 0.165,
            "TP +25% / DTE≤10일", color=C_MUTED, fontsize=9)
    ax.text(str_x + 0.02, box_y + box_h - 0.205,
            "트리거: natural_hedge,", color="#a78bfa", fontsize=8.5)
    ax.text(str_x + 0.02, box_y + box_h - 0.225,
            "crash_vulnerable,", color="#a78bfa", fontsize=8.5)
    ax.text(str_x + 0.02, box_y + box_h - 0.245,
            "co_crash_cluster,", color="#a78bfa", fontsize=8.5)
    ax.text(str_x + 0.02, box_y + box_h - 0.265,
            "fat_tail_alert (high_vix)", color="#a78bfa", fontsize=8.5)
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
            "베헤이클: ETF 현물 롱/숏", color=C_MUTED, fontsize=9)
    ax.text(dir_x + 0.02, box_y + box_h - 0.135,
            "예산: 5% × scale × mult", color=C_MUTED, fontsize=9)
    ax.text(dir_x + 0.02, box_y + box_h - 0.165,
            "TP +10% / SL -7% / 21일", color=C_MUTED, fontsize=9)
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
    ax.text(0.04, 0.215, "시스템 테제 코히어런스 (2026-06-10 라운드 4 (A) 완료)",
            color=C_ACCENT_4, fontsize=12, fontweight="bold")
    ax.text(0.04, 0.175,
            "현재 mid_vix:  LONG STRADDLE 0건  ·  DIRECTIONAL 6건 (rate)  ·  "
            "SHORT STRADDLE 0건",
            color=C_TEXT, fontsize=10, fontweight="bold")
    ax.text(0.04, 0.140,
            "→ 위기 룰 미발화 = 비싼 IV 매수 X  ·  그린라이트 미발화 = 꼬리 두꺼우니 매도 X",
            color=C_MUTED, fontsize=9.5)
    ax.text(0.04, 0.105,
            "→ 평상시엔 directional 만, 위기엔 LONG 자동 발사, 진짜 안전 시 SHORT 발사",
            color=C_ACCENT_3, fontsize=10, fontweight="bold")
    ax.text(0.04, 0.060,
            "feedback (rule × strategy) 키 학습 — MIN_TRADES=5, clip(0.7, 1.3), stop-rule n≥10 & avg<-5% → mult=0",
            color=C_MUTED, fontsize=9)
    ax.text(0.04, 0.025,
            "두 readout 부분 상관 (ρ ∈ (0,1)). '두 배 학습' 과장. 핵심 룰은 표본 누적 수년 — 사실상 열린 루프.",
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
    ax.text(0.04, 0.04,
            "제약: 옵션 마켓 주문은 RTH(09:30–16:00 ET) 만 가능. "
            "휴장 시 자동 가드. .env 자격증명 (코드 하드코딩 금지). "
            "피드백은 사이징 승수만 자동 — 임계값 T 와 청산 룰은 수동.",
            color=C_MUTED, fontsize=9.5)

    _footer(fig, n, total)
    pdf.savefig(fig, facecolor=C_BG); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Page — Backtest results (NEW 2026-06-10)
# ──────────────────────────────────────────────────────────────

def page_backtest(pdf, n, total):
    fig = _new_page(figsize=(12.0, 9.0))
    _title(fig, "백테스트 결과 — 6개월 (2025-12-10 ~ 2026-06-08)",
           "194 closed trades · BS+GARCH IV · 호가스프레드 5% 슬리피지 · look-ahead bias 한계 있음")

    ax = _blank_axes(fig, (0.04, 0.06, 0.92, 0.82))

    # 전략별 요약 박스
    box_y = 0.78
    box_h = 0.10
    box_w = 0.30

    strategies = [
        ("STRADDLE", 112, 52, -1.8, -1388, "#8b5cf6"),
        ("DIRECTIONAL", 82, 45, 0.0, -26, C_ACCENT_2),
        ("SHORT STRADDLE", 0, 0, 0, 0, C_ACCENT_3),
    ]
    for i, (name, n_t, win, avg, total_pnl, color) in enumerate(strategies):
        x = 0.02 + i * (box_w + 0.02)
        p = FancyBboxPatch((x, box_y), box_w, box_h,
                           boxstyle="round,pad=0.02,rounding_size=0.03",
                           linewidth=1.4, edgecolor=color, facecolor=C_PANEL)
        ax.add_patch(p)
        ax.text(x + box_w / 2, box_y + box_h - 0.025, name,
                ha="center", color=color, fontsize=11, fontweight="bold")
        if n_t > 0:
            ax.text(x + box_w / 2, box_y + box_h - 0.05,
                    f"n={n_t}  win={win}%", ha="center",
                    color=C_TEXT, fontsize=10)
            ax.text(x + box_w / 2, box_y + box_h - 0.075,
                    f"avg={avg:+.1f}%   total=${total_pnl:+,.0f}",
                    ha="center", color=C_ACCENT_5 if total_pnl < 0 else C_ACCENT_3,
                    fontsize=10, fontweight="bold")
        else:
            ax.text(x + box_w / 2, box_y + box_h / 2 - 0.005,
                    "트랙 신설 후 발사 0 — 아직 데이터 없음",
                    ha="center", color=C_MUTED, fontsize=9, style="italic")

    # (rule × strategy) 표
    ax.text(0.02, 0.62, "(rule × strategy) 정합성 검증",
            color=C_ACCENT_4, fontsize=12, fontweight="bold")

    rule_results = [
        ("co_crash_cluster | straddle", 8, 75, 20.4, 1077, True, "✓ 위기 룰 + 시스템 테제 직결"),
        ("fat_tail_alert | straddle",   4, 75, 9.4, 133, True, "✓ 표본 작지만 valid"),
        ("rate_beneficiary | directional", 14, 50, 3.2, 123, True, "✓ macro sensitivity 직접"),
        ("shock_propagator | directional", 33, 48, -0.4, -41, False, "≈ break-even"),
        ("rate_victim | directional", 29, 41, -0.5, -64, False, "약한 음수"),
        ("causal_chain_monitor | straddle", 50, 50, -4.3, -649, False, "❌ 누적 손실 → 제거됨"),
        ("shock_propagator | straddle", 50, 48, -3.9, -1949, False, "❌ 가장 큰 손실 → 제거됨"),
    ]
    # Header
    headers = [("키", 0.02), ("n", 0.32), ("win%", 0.37), ("avg%", 0.43),
               ("total $", 0.50), ("진단", 0.62)]
    y = 0.58
    for h, x in headers:
        ax.text(x, y, h, color=C_ACCENT_4, fontsize=10, fontweight="bold")

    y -= 0.035
    for key, n_t, win, avg, total_pnl, is_pos, diagnosis in rule_results:
        ax.axhline(y + 0.015, xmin=0.02, xmax=0.98, color=C_LIGHT, lw=0.4, alpha=0.3)
        row_color = C_ACCENT_3 if is_pos else (C_ACCENT_5 if "❌" in diagnosis else C_MUTED)
        ax.text(0.02, y, key, color=row_color, fontsize=9.5,
                fontweight="bold", fontfamily="monospace")
        ax.text(0.32, y, f"{n_t}", color=C_TEXT, fontsize=9.5)
        ax.text(0.37, y, f"{win}%", color=C_TEXT, fontsize=9.5)
        ax.text(0.43, y, f"{avg:+.1f}%", color=row_color, fontsize=9.5, fontweight="bold")
        ax.text(0.50, y, f"${total_pnl:+,.0f}", color=row_color, fontsize=9.5, fontweight="bold")
        ax.text(0.62, y, diagnosis, color=C_TEXT, fontsize=9.5)
        y -= 0.038

    # 핵심 결론
    ax.text(0.02, 0.205, "핵심 결론",
            color=C_ACCENT_4, fontsize=12, fontweight="bold")
    conclusions = [
        "① STRADDLE 전체 -$1,388 — 호가스프레드 + 세타가 시그널 알파보다 큼",
        "② 알파 만든 룰 = high_vix 발화 (co_crash_cluster, fat_tail_alert) + rate_beneficiary",
        "③ 비인과 휴리스틱 룰 명백한 실패 → 11 → 8 룰 축소의 직접 근거",
        "④ stop-rule 임계 변경 안 함 — in-sample 과적합 회피 (라운드 5 메타 비평)",
    ]
    y = 0.165
    for c in conclusions:
        ax.text(0.02, y, c, color=C_TEXT, fontsize=9.5, fontweight="normal")
        y -= 0.030

    # Look-ahead bias 경고
    ax.text(0.02, 0.025,
            "⚠ 알려진 한계: tail_gpd/co_crash 통계가 모든 과거 데이터로 한 번 적합 → look-ahead bias. "
            "co_crash_cluster +20.4% 가 진짜인지 부분 확신 X. GARCH IV ≠ 진짜 IV (상한 추정).",
            color=C_ACCENT_4, fontsize=9, style="italic")

    _footer(fig, n, total)
    pdf.savefig(fig, facecolor=C_BG); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Page 12 — Known limitations (외부 비평 반영)
# ──────────────────────────────────────────────────────────────

def page_limitations(pdf, n, total):
    fig = _new_page(figsize=(12.0, 9.0))
    _title(fig, "알려진 구조적 한계 (2026-06-10 갱신)",
           "라운드 4 비평 4 항목 중 4 해결 ✓. 남은 2 항목 = 구조적 한계.")

    ax = _blank_axes(fig, (0.04, 0.06, 0.92, 0.82))

    items = [
        ("0. ✓ VRP 측정 오류 정정 (라운드 4)",
         "Alpaca options snapshot 의 ATM 콜+풋 IV → VRP_true = ATM_IV - RV_20d.\n"
         "XLB σ_GARCH=-0.86% → IV 기반 +1.89% (부호 뒤집힘). 비평자 예측 실증.",
         C_ACCENT_3),

        ("1. ✓ 숏 볼 수단 부재 → 해결 (2026-06-10)",
         "rule_thin_tail_greenlight 신설: EVT(ξ<0.10) AND Copula(λ_L<0.20) AND CF<1.5.\n"
         "vol_overpriced ∩ thin_tail_greenlight 만 SHORT STRADDLE 발사. high_vix 자동 비활성.\n"
         "현재 mid_vix: 그린라이트 0건 → 자동 0 발사 (시스템 가설 정직).",
         C_ACCENT_3),

        ("2. ✓ 백테스트로 첫 P&L 산출 (라운드 4 (B) 우회)",
         "backtest_pipeline.py: 6개월 / 194 closed / STRADDLE -1.8% / co_crash +20.4% / shock_prop -3.9%.\n"
         "look-ahead bias 알려진 한계 (tail_gpd/co_crash 시점-aware 재적합 X).\n"
         "페이퍼 첫 청산은 여전히 0건 — 실거래 검증은 시간 필요.",
         C_ACCENT_3),

        ("3. ✓ 정합성 의심 룰 제거 (라운드 5 선반영)",
         "shock_propagator / variance_concentrated / causal_chain_monitor 제거 (11 → 8 룰).\n"
         "Cholesky/Shapley 분해는 *분석 도구*(UI)로만 — 룰 입력 X.\n"
         "stop-rule 임계 변경 없음 — in-sample 과적합 회피.",
         C_ACCENT_3),

        ("4. (남은 한계) 22 승수 노이즈, 핵심 룰 열린 루프",
         "MIN_TRADES=5, clip(0.7, 1.3), stop-rule n≥10 & avg<-5% 적용 중.\n"
         "high_vix 발화 룰은 표본 누적 수년 — 사실상 열린 루프 (구조적 한계).",
         C_ACCENT_1),

        ("5. (남은 한계) VQC macro 측 미검증, look-ahead bias",
         "VQC: invest-portal 에만 검증, macro 미통합. A/B 실험 미진행.\n"
         "백테스트의 tail_gpd/co_crash 통계 시점-aware 재적합 미구현 → +20.4% 진위 불확실.",
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

    ax.text(0.5, 0.005,
            "2026-06-10 기준: 라운드 4 비평 4개 항목 중 4개 해결, 남은 2개는 구조적 한계. "
            "다음은 시점-aware 재적합 + 실거래 검증 누적.",
            ha="center", color=C_ACCENT_3, fontsize=10, fontweight="bold")

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
            y -= 0.073

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
    total = 14
    with PdfPages(PDF_OUT) as pdf:
        page_cover(pdf, total)
        page_flow(pdf, 2, total)
        page_garch(pdf, 3, total)
        page_evt(pdf, 4, total)
        page_copula(pdf, 5, total)
        page_cholesky(pdf, 6, total, live)
        page_pcmci(pdf, 7, total)
        page_rules(pdf, 8, total)
        page_joint(pdf, 9, total, live)
        page_dual(pdf, 10, total)
        page_kinetic_loop(pdf, 11, total)
        page_backtest(pdf, 12, total)            # NEW 2026-06-10
        page_limitations(pdf, 13, total)
        page_glossary(pdf, 14, total)
        d = pdf.infodict()
        d["Title"]    = "Macro Research System Overview"
        d["Author"]   = "wjdrj"
        d["Subject"]  = "6-Layer 온톨로지 + 트리플 전략 + Kinetic Layer + 백테스트"
        d["Keywords"] = "GARCH EVT Copula Cholesky Shapley PCMCI NORTA Straddle Alpaca Backtest"

    print(f"[overview-pdf] 완료. 크기={PDF_OUT.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()

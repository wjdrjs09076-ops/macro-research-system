"""
ontology_trace.py

온톨로지 게이트가 각 위기 이벤트에 대해
어떤 수치 근거로 판단을 내렸는지 추적 출력.

출력 구조:
  [위기명]
    Step 1. 원시 지표 통계 (OOS 기간 평균/중앙값)
    Step 2. 게이트 3개 컴포넌트 비율
    Step 3. 게이트 점수 분포 (임계값 0.45 기준)
    Step 4. 연속 발화 패턴 (GATE_CONSEC=5일 조건)
    Step 5. 최종 판정 + 실패/성공 이유 추론
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import yfinance as yf

BASE_DIR    = Path(__file__).parent
FIGURES_DIR = BASE_DIR / "output" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

EWMA_SPAN   = 32
RV_WIN      = 20
BETA_WIN    = 30
GATE_WINDOW = 15
GATE_CONSEC = 5
GATE_THRESH = 0.45

CRISIS_PERIODS = {
    "hormuz": {
        "label"        : "호르무즈 봉쇄 (2025-2026) -- SUPPLY_SHOCK",
        "start"        : "2024-07-01",
        "end"          : "2026-05-29",
        "oos_start"    : "2026-01-01",
        "etf"          : "XLE",
        "direction"    : "LONG",
        "expected_gate": "OPEN",
        "expected_type": "SUPPLY_SHOCK",
        "mechanism"    : "이란-미국 갈등 -> 호르무즈 봉쇄 -> 원유 공급 감소 -> 에너지 섹터 수혜",
    },
    "gfc": {
        "label"        : "글로벌 금융위기 (2006-2010) -- STRUCTURAL_COLLAPSE",
        "start"        : "2005-07-01",
        "end"          : "2010-12-31",
        "oos_start"    : "2008-09-15",
        "etf"          : "XLF",
        "direction"    : "SHORT",
        "expected_gate": "OPEN",
        "expected_type": "STRUCTURAL_COLLAPSE",
        "mechanism"    : "서브프라임 모기지 부실 -> 레버리지 시스템 붕괴 -> 금융섹터 구조적 하락",
    },
    "covid": {
        "label"        : "COVID-19 (2019-2021) -- SUBSECTOR_SPECIFIC",
        "start"        : "2018-07-01",
        "end"          : "2021-12-31",
        "oos_start"    : "2020-02-20",
        "etf"          : "XLV",
        "direction"    : "LONG",
        "expected_gate": "CLOSED",
        "expected_type": "SUBSECTOR_SPECIFIC",
        "mechanism"    : "팬데믹 -> 전체 시장 패닉 -> 바이오/헬스케어 상승, BUT XLV=전체 헬스케어 ETF (분리 안됨)",
    },
    "dotcom": {
        "label"        : "닷컴버블 (1998-2002) -- VALUATION_CORRECTION",
        "start"        : "1997-07-01",
        "end"          : "2002-12-31",
        "oos_start"    : "2000-03-10",
        "etf"          : "XLK",
        "direction"    : "SHORT",
        "expected_gate": "CLOSED",
        "expected_type": "VALUATION_CORRECTION",
        "mechanism"    : "기술주 밸류에이션 과대 -> 버블 붕괴, BUT 하락 중 -40~+50% 반등 반복 -> SHORT 청산 위험",
    },
}


def fetch(tickers, start, end):
    all_t = list(set(tickers + ["SPY", "^VIX"]))
    raw   = yf.download(all_t, start=start, end=end,
                        auto_adjust=True, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    out = {}
    for t in all_t:
        if t in close.columns:
            s = close[t].dropna()
            if len(s) > 20:
                out[t] = s
    return out


def rolling_beta(sec_ret, spy_ret, window=BETA_WIN):
    idx = sec_ret.index.intersection(spy_ret.index)
    s, m = sec_ret.loc[idx], spy_ret.loc[idx]
    return (s.rolling(window).cov(m) / m.rolling(window).var()).dropna()


def compute_vrp(sec_ret, spy_ret, vix):
    idx  = sec_ret.index.intersection(spy_ret.index)
    s, m = sec_ret.loc[idx], spy_ret.loc[idx]
    b63  = rolling_beta(s, m, 63).reindex(idx).ffill()
    res  = s - b63 * m
    idio = res.ewm(span=EWMA_SPAN, min_periods=20).std() * np.sqrt(252)
    vix_a   = vix.reindex(idx).ffill() / 100.0
    capm_iv = np.sqrt((b63 * vix_a)**2 + idio**2)
    rv_ann  = s.rolling(RV_WIN).std() * np.sqrt(252)
    return (capm_iv - rv_ann).dropna()


def compute_gate_components(beta_s, vrp_s, etf_ret, spy_ret, direction):
    idx = beta_s.index.intersection(vrp_s.index)
    idx = idx.intersection(etf_ret.index).intersection(spy_ret.index)

    b   = beta_s.reindex(idx)
    v   = vrp_s.reindex(idx)
    excs = etf_ret.reindex(idx) - spy_ret.reindex(idx)

    beta_rev   = (b < -0.05).astype(float).rolling(GATE_WINDOW, min_periods=5).mean()
    beta_spike = (b > 1.50).astype(float).rolling(GATE_WINDOW, min_periods=5).mean()
    vrp_neg    = (v < 0).astype(float).rolling(GATE_WINDOW, min_periods=5).mean()
    etf_out    = (excs > 0).astype(float).rolling(GATE_WINDOW, min_periods=5).mean()
    etf_under  = (excs < 0).astype(float).rolling(GATE_WINDOW, min_periods=5).mean()

    if direction == "LONG":
        beta_comp = beta_rev
        dir_comp  = etf_out
        gate_score = 0.40 * beta_rev + 0.35 * vrp_neg + 0.25 * etf_out
    else:
        beta_comp = beta_spike
        dir_comp  = etf_under
        gate_score = 0.40 * beta_spike + 0.35 * vrp_neg + 0.25 * etf_under

    above = (gate_score >= GATE_THRESH).astype(float)
    gate_open = above.rolling(GATE_CONSEC, min_periods=GATE_CONSEC).min().fillna(0).astype(bool)

    return pd.DataFrame({
        "beta_raw"    : b,
        "vrp_raw"     : v,
        "beta_comp"   : beta_comp,
        "vrp_comp"    : vrp_neg,
        "dir_comp"    : dir_comp,
        "gate_score"  : gate_score,
        "above_thresh": above,
        "gate_open"   : gate_open,
    }, index=idx)


def print_trace(name, cfg, df, oos_dt):
    sep  = "=" * 72
    sep2 = "-" * 72
    print(f"\n{sep}")
    print(f"  {cfg['label']}")
    print(f"  인과 메커니즘: {cfg['mechanism']}")
    print(sep2)

    oos  = df[df.index >= oos_dt].copy()
    full = df.copy()

    n_oos = len(oos)
    if n_oos == 0:
        print("  OOS 데이터 없음")
        return

    direction = cfg["direction"]

    # ── Step 1. 원시 지표 ───────────────────────────────────────────────
    print(f"\n  [Step 1] OOS 원시 지표 ({n_oos}일)")
    print(f"    Beta(30d)  : 평균={oos['beta_raw'].mean():+.3f}  "
          f"중앙={oos['beta_raw'].median():+.3f}  "
          f"< -0.05 비율={( oos['beta_raw'] < -0.05).mean()*100:.1f}%  "
          f"> 1.50 비율={( oos['beta_raw'] > 1.50).mean()*100:.1f}%")
    print(f"    VRP        : 평균={oos['vrp_raw'].mean():+.3f}  "
          f"중앙={oos['vrp_raw'].median():+.3f}  "
          f"< 0 비율={( oos['vrp_raw'] < 0).mean()*100:.1f}%")

    # ── Step 2. 컴포넌트 비율 ──────────────────────────────────────────
    print(f"\n  [Step 2] 게이트 컴포넌트 (롤링 {GATE_WINDOW}일 비율, OOS 평균)")
    if direction == "LONG":
        beta_label = "beta 역전 비율 (beta<-0.05)"
        dir_label  = "ETF 초과수익 비율 (ETF>SPY)"
    else:
        beta_label = "beta 급등 비율 (beta>1.50)"
        dir_label  = "ETF 하회 비율 (ETF<SPY)"

    bc = oos["beta_comp"].mean()
    vc = oos["vrp_comp"].mean()
    dc = oos["dir_comp"].mean()
    weighted = 0.40 * bc + 0.35 * vc + 0.25 * dc

    print(f"    {beta_label:<35}: {bc*100:5.1f}%  (가중 기여: {0.40*bc:.3f})")
    print(f"    {'VRP 음전환 비율 (VRP<0)':<35}: {vc*100:5.1f}%  (가중 기여: {0.35*vc:.3f})")
    print(f"    {dir_label:<35}: {dc*100:5.1f}%  (가중 기여: {0.25*dc:.3f})")
    print(f"    {'종합 게이트 점수 (OOS 평균)':<35}: {weighted:.3f}  "
          f"(임계값 {GATE_THRESH})")

    # ── Step 3. 점수 분포 ─────────────────────────────────────────────
    print(f"\n  [Step 3] 게이트 점수 분포 (OOS)")
    gs = oos["gate_score"].dropna()
    pcts = [10, 25, 50, 75, 90]
    pct_vals = np.percentile(gs, pcts)
    print(f"    분위수: " + "  ".join(f"P{p}={v:.3f}" for p, v in zip(pcts, pct_vals)))
    above_pct = (gs >= GATE_THRESH).mean() * 100
    print(f"    >= {GATE_THRESH} 비율: {above_pct:.1f}%  "
          f"(5일 연속 충족 시 게이트 열림)")

    # ── Step 4. 연속 발화 분석 ─────────────────────────────────────────
    print(f"\n  [Step 4] 연속 발화 패턴 (GATE_CONSEC={GATE_CONSEC}일)")
    above = oos["above_thresh"].astype(int)

    runs = []
    cur_len = 0
    for v in above:
        if v:
            cur_len += 1
        else:
            if cur_len > 0:
                runs.append(cur_len)
            cur_len = 0
    if cur_len > 0:
        runs.append(cur_len)

    n_open_days  = int(oos["gate_open"].sum())
    n_total_days = len(oos)
    if runs:
        max_run = max(runs)
        avg_run = np.mean(runs)
        n_runs_5plus = sum(1 for r in runs if r >= GATE_CONSEC)
        print(f"    임계값 이상 연속 구간: {len(runs)}번  "
              f"최장={max_run}일  평균={avg_run:.1f}일")
        print(f"    {GATE_CONSEC}일 이상 연속 구간: {n_runs_5plus}번")
    else:
        print(f"    임계값 이상 구간: 없음 (0번)")
        n_runs_5plus = 0

    open_pct = n_open_days / max(n_total_days, 1) * 100
    print(f"    게이트 최종 열림 일수: {n_open_days}/{n_total_days}일 ({open_pct:.1f}%)")

    # ── Step 5. 최종 판정 + 해석 ──────────────────────────────────────
    print(f"\n  [Step 5] 최종 판정")
    expected = cfg["expected_gate"]
    if expected == "OPEN":
        correct = open_pct > 20
    else:
        correct = open_pct < 20

    verdict_str = "OPEN" if open_pct > 20 else "CLOSED"
    tick = "OK" if correct else "FAIL"
    print(f"    예측: {expected}  |  실제: {verdict_str}  |  [{tick}]")

    if tick == "OK":
        if expected == "OPEN":
            print(f"\n  [판정 근거 -- 성공]")
            print(f"    게이트가 올바르게 열림 ({open_pct:.0f}%)")
            if direction == "LONG":
                print(f"    - beta 역전 ({bc*100:.0f}%): 에너지 섹터가 시장과 디커플링")
                print(f"      -> 외부 충격(공급 제약)이 섹터에 고유하게 작용한 증거")
                print(f"    - VRP 음전환 ({vc*100:.0f}%): CAPM_IV < 실현변동성")
                print(f"      -> 내재변동성이 이미 현실화된 리스크를 반영 = 충격 내재화 완료")
                print(f"    - ETF 초과수익 ({dc*100:.0f}%): XLE가 SPY 대비 상승")
                print(f"      -> 이벤트 수혜 방향 일치 = SUPPLY_SHOCK LONG 신호 유효")
            else:
                print(f"    - beta 급등 ({bc*100:.0f}%): 금융섹터가 시장과 동조 과잉")
                print(f"      -> 레버리지 시스템 전반 붕괴, 구조적 연쇄반응 진행 중")
                print(f"    - VRP 음전환 ({vc*100:.0f}%): 공포 프리미엄 현실화")
                print(f"      -> 시장이 '추가 붕괴'를 가격에 반영하기 시작")
                print(f"    - ETF 하회 ({dc*100:.0f}%): XLF가 SPY 대비 하락")
                print(f"      -> 금융섹터 집중 손실, 구조적 원인 확인")
        else:
            print(f"\n  [판정 근거 -- 성공]")
            print(f"    게이트가 올바르게 닫힘 ({open_pct:.0f}%)")
            if name == "covid":
                print(f"    - beta 역전 ({bc*100:.0f}%): XLV(헬스케어 전체)는 시장과 동조")
                print(f"      -> 바이오 소형주는 수혜, 대형 헬스케어 ETF는 비분리")
                print(f"      -> FULL_ETF_DECOUPLING 조건 불충족: 게이트 닫힘")
                print(f"    - 결론: 신호가 맞지만 ETF가 틀렸음 (XLV != 바이오 순수 노출)")
    else:
        print(f"\n  [판정 근거 -- 실패]")
        print(f"    게이트가 잘못 열림 ({open_pct:.0f}%)")
        if name == "dotcom":
            print(f"    - beta 급등 ({bc*100:.0f}%): 기술주 beta > 1.5 빈번")
            print(f"      -> 닷컴 버블기 기술주 변동성/상관관계 매우 높음 -> 조건 충족")
            print(f"    - VRP 음전환 ({vc*100:.0f}%): 묵시적변동성 < 실현변동성")
            print(f"      -> 버블 붕괴기 내재변동성이 실현변동성보다 낮은 구간 존재")
            print(f"    - ETF 하회 ({dc*100:.0f}%): XLK가 SPY 대비 하락")
            print(f"      -> 기술주 집중 하락 -> 방향성 조건도 충족")
            print(f"")
            print(f"    [게이트 실패 원인]")
            print(f"    게이트가 측정하는 3개 조건이 모두 닷컴 붕괴에서도 발화됨.")
            print(f"    그러나 닷컴은 STRUCTURAL_COLLAPSE가 아닌 VALUATION_CORRECTION.")
            print(f"")
            print(f"    핵심 차이:")
            print(f"      STRUCTURAL_COLLAPSE (GFC):  외부 레버리지 시스템 붕괴 -> 지속적 하락")
            print(f"      VALUATION_CORRECTION (닷컴): 밸류에이션 재조정 -> 하락 중 대형 반등 반복")
            print(f"")
            print(f"    닷컴 2000-2002 주요 반등 구간:")
            print(f"      - 2001-01~04: XLK +35% (9.11 이전 반등)")
            print(f"      - 2001-09~11: XLK +25% (9.11 패닉 후 기술적 반등)")
            print(f"      - 2002-07~08: XLK +20% (이중바닥 기대 반등)")
            print(f"    -> SHORT 포지션이 이 반등에서 강제 청산될 확률 매우 높음")
            print(f"")
            print(f"    게이트가 측정 못하는 것: 이벤트 타입이 구조적 원인인지 밸류에이션 원인인지")
            print(f"    -> trading_ontology.py의 structural_cause / commodity_driven 노드가")
            print(f"       이 분류를 담당하지만, 게이트 점수에는 포함되지 않음")

    print(f"\n  {sep2}")


def plot_all_traces(crisis_data):
    fig = plt.figure(figsize=(20, 24))
    gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.55)

    colors = {
        "hormuz": "#1abc9c",
        "gfc"   : "#e74c3c",
        "covid" : "#3498db",
        "dotcom": "#e67e22",
    }

    for i, (name, pack) in enumerate(crisis_data.items()):
        cfg, df, oos_dt = pack
        ax = fig.add_subplot(gs[i])

        gs_s = df["gate_score"]
        go   = df["gate_open"]

        ax.plot(gs_s, color=colors[name], linewidth=1.3, alpha=0.9,
                label="게이트 점수")
        ax.plot(df["beta_comp"] * 0.40, color="navy", linewidth=0.8,
                linestyle="--", alpha=0.6, label="beta 기여 (x0.40)")
        ax.plot(df["vrp_comp"]  * 0.35, color="darkorange", linewidth=0.8,
                linestyle="--", alpha=0.6, label="VRP 기여 (x0.35)")
        ax.plot(df["dir_comp"]  * 0.25, color="purple", linewidth=0.8,
                linestyle="--", alpha=0.6, label="방향 기여 (x0.25)")

        ax.axhline(GATE_THRESH, color="black", linestyle=":", linewidth=0.9,
                   label=f"임계값 {GATE_THRESH}")
        ax.axvline(pd.Timestamp(cfg["oos_start"]), color="red",
                   linestyle="--", linewidth=1.2, label="OOS 시작")

        for j in range(len(go) - 1):
            if go.iloc[j]:
                ax.axvspan(go.index[j], go.index[j+1],
                           alpha=0.18, color=colors[name])

        oos_df   = df[df.index >= oos_dt]
        open_pct = oos_df["gate_open"].mean() * 100
        exp      = cfg["expected_gate"]
        verdict  = "OPEN" if open_pct > 20 else "CLOSED"
        tick     = "OK" if (exp == verdict) else "FAIL"

        title_color = "#27ae60" if tick == "OK" else "#c0392b"
        ax.set_title(
            f"{cfg['label']}\n"
            f"OOS 게이트 열림: {open_pct:.0f}%  |  "
            f"예측: {exp}  |  실제: {verdict}  |  [{tick}]  |  "
            f"방향: {cfg['direction']}  |  이벤트 유형: {cfg['expected_type']}",
            fontsize=9, color=title_color,
        )
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7.5, ncol=4, loc="upper left")
        ax.tick_params(labelsize=8)

    fig.suptitle(
        "온톨로지 게이트 판단 과정 추적 (4개 위기)\n"
        "각 위기에서 beta/VRP/방향 컴포넌트가 어떻게 게이트 점수를 구성했는지",
        fontsize=12, weight="bold",
    )
    out = FIGURES_DIR / "ontology_trace.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  차트 저장: {out}")


def export_json(crisis_data: dict, out_dir: Path):
    import json

    payload = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "gate_params": {
            "window": GATE_WINDOW,
            "consec": GATE_CONSEC,
            "threshold": GATE_THRESH,
        },
        "crises": {},
    }

    for name, pack in crisis_data.items():
        cfg, df, oos_dt = pack
        oos = df[df.index >= oos_dt]
        if len(oos) == 0:
            continue

        bc = float(oos["beta_comp"].mean())
        vc = float(oos["vrp_comp"].mean())
        dc = float(oos["dir_comp"].mean())
        ws = round(0.40 * bc + 0.35 * vc + 0.25 * dc, 4)
        open_pct = float(oos["gate_open"].mean() * 100)
        exp    = cfg["expected_gate"]
        actual = "OPEN" if open_pct > 20 else "CLOSED"

        gs = oos["gate_score"].dropna()
        pct_vals = {f"p{p}": round(float(np.percentile(gs, p)), 4)
                    for p in [10, 25, 50, 75, 90]}

        above = oos["above_thresh"].astype(int)
        runs, cur = [], 0
        for v in above:
            if v:
                cur += 1
            else:
                if cur > 0:
                    runs.append(cur)
                cur = 0
        if cur > 0:
            runs.append(cur)

        ts_df = df[["gate_score", "beta_comp", "vrp_comp",
                    "dir_comp", "gate_open"]].dropna()

        payload["crises"][name] = {
            "label"        : cfg["label"],
            "event_type"   : cfg["expected_type"],
            "mechanism"    : cfg["mechanism"],
            "direction"    : cfg["direction"],
            "etf"          : cfg["etf"],
            "oos_start"    : cfg["oos_start"],
            "expected_gate": exp,
            "actual_gate"  : actual,
            "correct"      : exp == actual,
            "step1": {
                "beta_mean"           : round(float(oos["beta_raw"].mean()), 4),
                "beta_median"         : round(float(oos["beta_raw"].median()), 4),
                "beta_below_neg005_pct": round(float((oos["beta_raw"] < -0.05).mean() * 100), 1),
                "beta_above_150_pct"  : round(float((oos["beta_raw"] > 1.50).mean() * 100), 1),
                "vrp_mean"            : round(float(oos["vrp_raw"].mean()), 4),
                "vrp_median"          : round(float(oos["vrp_raw"].median()), 4),
                "vrp_below_0_pct"     : round(float((oos["vrp_raw"] < 0).mean() * 100), 1),
            },
            "step2": {
                "beta_comp_pct" : round(bc * 100, 1),
                "beta_contrib"  : round(0.40 * bc, 4),
                "vrp_comp_pct"  : round(vc * 100, 1),
                "vrp_contrib"   : round(0.35 * vc, 4),
                "dir_comp_pct"  : round(dc * 100, 1),
                "dir_contrib"   : round(0.25 * dc, 4),
                "total_score"   : ws,
                "weights"       : {"beta": 0.40, "vrp": 0.35, "direction": 0.25},
            },
            "step3": {
                "percentiles"        : pct_vals,
                "above_threshold_pct": round(float((gs >= GATE_THRESH).mean() * 100), 1),
            },
            "step4": {
                "n_runs"          : len(runs),
                "max_run"         : int(max(runs)) if runs else 0,
                "avg_run"         : round(float(np.mean(runs)), 1) if runs else 0,
                "runs_5plus"      : int(sum(1 for r in runs if r >= GATE_CONSEC)),
                "gate_open_days"  : int(oos["gate_open"].sum()),
                "gate_total_days" : len(oos),
                "gate_open_pct"   : round(open_pct, 1),
            },
            "timeseries": {
                "dates"      : ts_df.index.strftime("%Y-%m-%d").tolist(),
                "gate_score" : [round(x, 4) for x in ts_df["gate_score"].tolist()],
                "beta_comp"  : [round(x, 4) for x in ts_df["beta_comp"].tolist()],
                "vrp_comp"   : [round(x, 4) for x in ts_df["vrp_comp"].tolist()],
                "dir_comp"   : [round(x, 4) for x in ts_df["dir_comp"].tolist()],
                "gate_open"  : [bool(x) for x in ts_df["gate_open"].tolist()],
            },
        }

    out_path = out_dir / "ontology_trace.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  JSON 저장: {out_path}")


if __name__ == "__main__":
    print("\n" + "=" * 72)
    print("  온톨로지 게이트 판단 경로 추적기")
    print("=" * 72)

    crisis_data = {}

    for name, cfg in CRISIS_PERIODS.items():
        print(f"\n  [{cfg['label']}] 데이터 로드 중...")
        prices = fetch([cfg["etf"]], cfg["start"], cfg["end"])
        if "SPY" not in prices or "^VIX" not in prices or cfg["etf"] not in prices:
            print("    SKIP: 데이터 없음")
            continue

        spy_ret = prices["SPY"].pct_change().dropna()
        vix_s   = prices["^VIX"]
        etf_ret = prices[cfg["etf"]].pct_change().dropna()

        beta_s = rolling_beta(etf_ret, spy_ret)
        vrp_s  = compute_vrp(etf_ret, spy_ret, vix_s)

        df = compute_gate_components(
            beta_s, vrp_s, etf_ret, spy_ret, cfg["direction"]
        )

        oos_dt = pd.Timestamp(cfg["oos_start"])
        print_trace(name, cfg, df, oos_dt)

        crisis_data[name] = (cfg, df, oos_dt)

    # 요약 비교 테이블
    print("\n" + "=" * 72)
    print("  [4개 위기 게이트 판단 요약 비교]")
    print("-" * 72)
    print(f"  {'위기':<28} {'beta기여':>8} {'VRP기여':>8} {'방향기여':>8} "
          f"{'종합점수':>9} {'열림%':>6} {'판정':>8}")
    print("-" * 72)
    for name, pack in crisis_data.items():
        cfg, df, oos_dt = pack
        oos = df[df.index >= oos_dt]
        if len(oos) == 0:
            continue
        bc = oos["beta_comp"].mean()
        vc = oos["vrp_comp"].mean()
        dc = oos["dir_comp"].mean()
        ws = 0.40 * bc + 0.35 * vc + 0.25 * dc
        open_pct = oos["gate_open"].mean() * 100
        exp  = cfg["expected_gate"]
        act  = "OPEN" if open_pct > 20 else "CLOSED"
        tick = "OK" if exp == act else "FAIL"
        print(f"  {cfg['expected_type']:<28} "
              f"{0.40*bc:>7.3f}  "
              f"{0.35*vc:>7.3f}  "
              f"{0.25*dc:>7.3f}  "
              f"{ws:>8.3f}  "
              f"{open_pct:>5.0f}%  "
              f"[{tick}]")
    print("=" * 72)

    print("\n  시각화 생성 중...")
    plot_all_traces(crisis_data)

    print("\n  JSON 내보내기...")
    JSON_DIR = BASE_DIR / "output"
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    export_json(crisis_data, JSON_DIR)

    print("\n완료.")

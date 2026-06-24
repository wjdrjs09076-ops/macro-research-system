"""
attribution.py - 거래 저널을 룰/레짐 단위로 귀인 분석.

trade_journal.closed_trades() (진입↔청산 조인)를 읽어
  - 룰별: 거래수, 승률, 평균 P&L%, 총 P&L, 평균 보유일
  - 레짐별: 동일 지표
를 계산하고 output/attribution.json 에 저장 + 표 출력.

귀인은 primary rule(진입 시 대표 룰) 기준. 같은 진입에 여러 룰이 함께
발화한 경우 all_rules 기준 분해는 향후 과제(현재는 대표 룰만 집계).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUTPUT_DIR
from trade_journal import closed_trades

ATTRIBUTION_JSON = OUTPUT_DIR / "attribution.json"

# 전향 검증 패널(Gate Timeline)용 export — 라이브 코드 불간섭, 거래 저널만 읽음(freeze 무관).
PORTAL_DATA_DIR = OUTPUT_DIR.parent.parent / "macro-portal" / "public" / "data"
FWD_JSON = PORTAL_DATA_DIR / "forward_validation.json"
OOS_FORWARD_START = "2026-06-16"        # freeze v2 = β헤지 페어 라이브 시작
SLIPPAGE_ROUNDTRIP = 0.54               # 27%×2 (n=2 실측 추정). 표본 누적 시 실측 중앙값으로 갱신.
CLUSTER_MIN_N = 8                        # 군집 판정 게이트 (이 미만이면 보류)
# event_vol z≥2.5 임계가 2026-06-15 origin-event(이란 휴전) 표본을 보고 선택됨(R9, 2.0/2.5/3.0 중
# 2.5, 커밋 6/16 = 이벤트 직후) = 임계 튜닝 in-sample. 따라서 그 표본기간의 모든 event_vol 발화
# (XLB 6/15 진입오염 + XLRE 6/18 임계오염)는 검증 제외. 진짜 clean = freeze 후 임계 고정된 채
# '완전히 새 이벤트'에서 z 뚫는 표본 — 자동탐지 불가, 그 이벤트 확인 시 아래 날짜로 설정.
# None = 전 event_vol 표본 검증 제외(clean n=0).
EVENT_VOL_CLEAN_AFTER = None


def export_forward_validation() -> dict:
    """전향 표본(닫힌 거래) → forward_validation.json (포털 패널 소스).

    필터: freeze v2(2026-06-16) 이후 청산 + 마이그레이션('구조전환(페어화)') 제외 =
    깨끗한 전향 표본만. 미실현/사전 정리 거래는 포함 안 함. 영점=슬리피지 비용대.
    """
    import datetime as dt
    trades = closed_trades()
    fwd = []
    for t in trades:
        ex = (t.get("exit_ts") or "")[:10]
        if not ex or ex < OOS_FORWARD_START:
            continue
        if t.get("exit_reason") == "구조전환(페어화)":   # 인프라 마이그레이션, thesis 청산 아님
            continue
        ent = (t.get("entry_ts") or "")[:10]
        validation, ex_reason = "clean", None
        if t.get("rule") == "event_vol" and t.get("strategy") == "straddle":
            if EVENT_VOL_CLEAN_AFTER is None or ent < EVENT_VOL_CLEAN_AFTER:
                validation = "excluded"
                ex_reason = ("event_vol 임계(z≥2.5)가 origin-event(6/15) 표본으로 선택됨 = 임계 튜닝 in-sample"
                             + (" + origin-event 진입 오염" if ent <= "2026-06-16" else ""))
        fwd.append({
            "exit_date":    ex,
            "entry_date":   ent,
            "rule":         t.get("rule"),
            "strategy":     t.get("strategy"),
            "ticker":       t.get("ticker"),
            "regime":       t.get("regime"),
            "pnl":          round(float(t.get("pnl") or 0), 2),
            "entry_cost":   round(float(t.get("entry_cost") or 0), 2),
            "exit_reason":  t.get("exit_reason"),
            "holding_days": t.get("holding_days"),
            "validation":   validation,        # clean | excluded
            "exclude_reason": ex_reason,
        })
    fwd.sort(key=lambda r: r["exit_date"])
    clean_n = sum(1 for r in fwd if r["validation"] == "clean")

    # ── churn 진단 (~6/30 히스테리시스 판정용 측정) ──────────────────────────
    # event_vol 임계(z≥2.5)가 경계에서 깜빡이면(2.51↔2.49) 매 사이클 진입↔은퇴 반복 →
    # 같은 종목 당일 진입+청산(holding_days==0) = 왕복 슬리피지만 까먹는 churn 왕복.
    # 레짐엔 히스테리시스(2사이클 확인)가 있지만 룰 발화엔 없음 → 이 깜빡임 미차단.
    # 측정 목적(오너 지시): "기록만" 말고 두 숫자를 누적해 6/30에 히스테리시스 정당성을
    # 데이터로 판정 — (a) churn 누적 비용, (b) 히스테리시스가 막았을 깜빡임 수.
    # 조작화: 0일 보유 event_vol 스트래들 왕복 = 깜빡임 1회(프록시). clean 표본 패배 시
    # "테제 오류 vs churn 슬리피지" 분해에도 이 비용을 빼는 기준선으로 쓴다.
    churn = [t for t in trades
             if t.get("rule") == "event_vol" and t.get("strategy") == "straddle"
             and t.get("holding_days") == 0]
    churn_pnl = round(sum(float(t.get("pnl") or 0) for t in churn), 2)

    payload = {
        "generated":          dt.datetime.now().isoformat(timespec="seconds"),
        "oos_forward_start":  OOS_FORWARD_START,
        "slippage_zero": {
            "roundtrip_pct": SLIPPAGE_ROUNDTRIP,
            "basis": "27%×2 고정 (n=2 실측 추정: XLB 28%/XLRE 27%). 표본 누적 시 실측 슬리피지 중앙값으로 갱신 예정.",
        },
        "cluster_min_n":      CLUSTER_MIN_N,
        "n_closed":           len(fwd),
        "clean_n":            clean_n,
        "excluded_n":         len(fwd) - clean_n,
        "boundary_note":      "event_vol 임계(z≥2.5)가 6/15 origin-event 표본으로 정해짐 → 그 표본기간 "
                              "event_vol 전부 검증 제외(임계 튜닝). 진짜 clean = freeze 후 새 이벤트 발화부터.",
        "timing_note":        "초기 표본은 cron 스로틀(외부 핑거 미설정)로 청산 타이밍이 신호 사망 시점과 "
                              "어긋날 수 있음(6/22 청산: 직전 사이클 2h42m 공백). 핑거 셋업 후 표본만 타이밍 clean.",
        "churn_diagnostic": {
            "definition":   "event_vol 스트래들 당일진입+청산(holding_days=0) = 임계경계 깜빡임 왕복. "
                            "룰 발화 히스테리시스 부재로 z가 2.5 경계서 깜빡이면 매 사이클 진입↔은퇴.",
            "flickers":     len(churn),          # (b) 히스테리시스가 막았을 깜빡임 수(프록시)
            "cum_pnl":      churn_pnl,           # (a) churn 누적 슬리피지 비용(음수=출혈)
            "verdict_note": "~6/30 판정용. flickers↑·cum_pnl↓ 면 히스테리시스 추가 정당성. "
                            "단 발화빈도를 바꾸므로 '구현 정합화'인지 '검증 대상 변경'인지는 오너 판정.",
        },
        "closed_trades":      fwd,
    }
    try:
        PORTAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
        FWD_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                       allow_nan=False), encoding="utf-8")
    except OSError as exc:
        print(f"  [WARN] forward_validation export 실패: {exc}")
    return payload


def _entry_date(t: dict) -> str:
    """진입 날짜 (cohort 키용). entry_ts 우선, 없으면 trade_id 의 날짜 부분."""
    ts = t.get("entry_ts")
    if ts:
        return str(ts)[:10]
    parts = (t.get("trade_id") or "").split("-")
    if len(parts) >= 4:
        return "-".join(parts[-4:-1])   # TICKER-YYYY-MM-DD-uuid → YYYY-MM-DD
    return "?"


def _agg(trades: list[dict], key, label: str | None = None) -> dict[str, dict]:
    """key 가 문자열이면 해당 필드로, 호출가능이면 t->버킷키 로 집계.

    cohort 통계 (2026-06-12): 같은 (rule, strategy, 진입일) 동시 진입은 같은
    매크로 드라이버로 함께 움직이는 *상관 표본* — 독립 증거로 세면 표본 수가
    부풀려진다 (금리 한 번 움직임 = rate_* 거래 6건 = "증거 6개" 왜곡).
    cohort 당 1 유효 표본으로 묶은 _eff 지표를 함께 산출한다.
    """
    keyfn = key if callable(key) else (lambda t, k=key: t.get(k) or "?")
    buckets: dict[str, list[dict]] = {}
    for t in trades:
        k = keyfn(t)
        buckets.setdefault(k, []).append(t)

    out: dict[str, dict] = {}
    for k, ts in buckets.items():
        pnls = [float(t.get("pnl") or 0) for t in ts]
        pnl_pcts = [float(t.get("pnl_pct") or 0) for t in ts]
        holds = [t["holding_days"] for t in ts if t.get("holding_days") is not None]
        wins = sum(1 for p in pnls if p > 0)
        n = len(ts)

        cohorts: dict[tuple, list[dict]] = {}
        for t in ts:
            ck = (t.get("rule"), t.get("strategy"), _entry_date(t))
            cohorts.setdefault(ck, []).append(t)
        co_mean_pcts = [
            sum(float(x.get("pnl_pct") or 0) for x in cts) / len(cts)
            for cts in cohorts.values()
        ]
        co_total_pnls = [
            sum(float(x.get("pnl") or 0) for x in cts)
            for cts in cohorts.values()
        ]
        n_eff = len(cohorts)

        out[k] = {
            "trades":          n,
            "wins":            wins,
            "win_rate":        round(wins / n, 4) if n else 0.0,
            "avg_pnl_pct":     round(sum(pnl_pcts) / n, 4) if n else 0.0,
            "total_pnl":       round(sum(pnls), 2),
            "avg_hold_days":   round(sum(holds) / len(holds), 1) if holds else None,
            "cohorts":         n_eff,
            "win_rate_eff":    round(sum(1 for p in co_total_pnls if p > 0) / n_eff, 4) if n_eff else 0.0,
            "avg_pnl_pct_eff": round(sum(co_mean_pcts) / n_eff, 4) if n_eff else 0.0,
        }
    return out


def _rule_strategy_key(t: dict) -> str:
    """feedback/사이징과 동일한 합성키 'rule|strategy'."""
    return f"{t.get('rule') or '?'}|{t.get('strategy') or 'straddle'}"


def build_attribution(write: bool = True) -> dict:
    trades = closed_trades()
    report = {
        "n_closed":         len(trades),
        "by_strategy":      _agg(trades, "strategy"),
        "by_rule_strategy": _agg(trades, _rule_strategy_key),
        "by_rule":          _agg(trades, "rule"),
        "by_regime":        _agg(trades, "regime"),
        "by_signal":        _agg(trades, "signal_type"),
    }
    if write:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ATTRIBUTION_JSON.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _print_table(title: str, stats: dict[str, dict]) -> None:
    print(f"\n  {title}")
    print(f"  {'key':32s} {'n':>4} {'n_eff':>5} {'win%':>6} {'avgP&L%':>9} "
          f"{'eff%':>7} {'totP&L$':>10} {'hold':>6}")
    print(f"  {'-'*84}")
    for k, s in sorted(stats.items(), key=lambda kv: -kv[1]["total_pnl"]):
        hold = f"{s['avg_hold_days']:.1f}" if s["avg_hold_days"] is not None else "-"
        print(f"  {k:32s} {s['trades']:>4} {s['cohorts']:>5} {s['win_rate']*100:>5.0f}% "
              f"{s['avg_pnl_pct']*100:>8.1f}% {s['avg_pnl_pct_eff']*100:>6.1f}% "
              f"{s['total_pnl']:>10,.0f} {hold:>6}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    report = build_attribution()
    print(f"\n{'='*64}")
    print(f"  귀인 분석 (ATTRIBUTION)  |  완결 거래 {report['n_closed']}건")
    print(f"{'='*64}")

    if report["n_closed"] == 0:
        print("\n  아직 완결된 거래 없음. 진입+청산이 한 쌍 이상 쌓이면 집계됩니다.")
        print(f"\n-> {ATTRIBUTION_JSON}")
        return

    _print_table("전략별 (strategy)", report["by_strategy"])
    _print_table("룰×전략별 (rule|strategy)", report["by_rule_strategy"])
    _print_table("룰별 (rule)", report["by_rule"])
    _print_table("레짐별 (regime)", report["by_regime"])
    _print_table("시그널타입별 (signal_type)", report["by_signal"])
    print(f"\n-> {ATTRIBUTION_JSON}")


if __name__ == "__main__":
    main()

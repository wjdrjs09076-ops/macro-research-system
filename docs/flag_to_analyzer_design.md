# 설계 메모 — Vol Monitor 플래그 → 이벤트 분석기 자동 인입 (감사 P2-10)

작성: 2026-06-13 · 상태: 설계만 (구현은 다음 라운드, freeze 영향 없음 — 평가 인프라)

## 문제

이벤트 분석기는 사용자가 *알려진 위기 이름*을 입력하는 구조 = 사후 선택(post-selection).
거기서 나오는 게이트 성과는 "사건을 미리 알았다면"의 조건부 수치라 전향 예측력의 증거가 아니다.
사후선택의 유일한 해독제는 **전향 파이프라인**: 시스템이 스스로 플래그한 시점부터 추적.

## 설계

```
vol_monitor_pipeline (30분 사이클)
  └─ etf_signals 중 level ≥ 2 (ALERT/HIGH) 발생
       └─ output/flag_log.jsonl append          ← P2-9 (구현됨)
            {ts, ticker, level, z_score, vrp, reason}
       └─ [다음 라운드] classify 자동 호출:
            event_name = f"{ticker} volatility spike {date}"  (이름 아닌 관측 기반)
            → quantify(start=flag-18mo, oos_start=flag일)    ← 사후선택 없는 oos_start
            → 결과를 flag_log 에 병기 (gate_open_pct, 이후 21일 실현 변동/수익)
  └─ 포털 /monitor 에 월별 패널:
       플래그 n / 게이트 OPEN 전환율 / 21일 내 |수익|>5% 비율 (이벤트 확정 proxy)
       — n 작음은 그대로 표기, 비율만으로 주장 금지
```

## 핵심 원칙

- **oos_start = 플래그 발생일** (사람이 고른 사건일이 아님) — designer leakage 차단.
- 플래그가 틀린 경우(전환율 낮음)도 그대로 게시 — 기존 불리한 결과 게시 원칙.
- 분석기 UI에는 "전향 모드" 탭으로 flag_log 항목을 읽기 전용 인입 (수동 입력과 시각 구분).

## 완료 기준 (다음 라운드)

flag_log 기반 자동 quantify 1회 사이클 통과 + /monitor 월별 hit-rate 패널 + 분석기 전향 모드 탭.

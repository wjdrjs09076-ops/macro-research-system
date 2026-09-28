# Macro Research System

**거시 지표와 섹터 ETF의 변동성 변화를 수집하고, 꼬리 위험·동조화·레짐을 점검해 해석 가능한 모니터링 신호로 정리하는 리서치 시스템입니다.**

[매크로 포털](https://macro-portal.vercel.app) · [상세 명세서](명세서.md) · [자동화 워크플로](.github/workflows/vol_monitor.yml)

## 리서치 질문과 흐름

시장 충격의 방향과 시점을 단정하기보다, 변동성 확대와 공동 하락 위험을 사전에 관찰할 수 있는지 묻습니다.

`시장·거시 데이터 → ETF 변동성 모니터 → GARCH/IV·EVT/GPD·꼬리 동조화 분석 → 온톨로지 규칙 → 기록·포털` 순서로 처리합니다. 시장 데이터에는 yfinance의 ETF·VIX, EIA 에너지 지표, FRED 거시 시계열을 사용합니다. 일부 분석에는 Nasdaq Data Link 자료와 Alpaca 옵션·페이퍼 계좌 연동이 필요합니다. 데이터별 접근 권한과 갱신 시점은 다릅니다.

## 구현한 일

- `vol_monitor_pipeline.py`: ETF 가격, VIX, 에너지 지표 등을 갱신하고 변동성 이상치를 포털용 JSON으로 만듭니다.
- `macro_research/tail_analysis.py`: GPD 꼬리지수와 손실 꼬리 지표, 경험적 하방·상방 동조화를 계산합니다. `ontology/`는 지표, 섹터, 규칙, 신호의 관계와 판단 근거를 기록합니다.
- `macro_research/causal_macro.py`: PCMCI를 이용해 거시 시계열 간 **후보 지연 관계**를 탐색합니다. 관찰 데이터에서 얻은 링크는 인과 효과의 확증이 아니며, 이를 섹터 수익률의 확정적 전파 경로로 해석하지 않습니다.
- `.github/workflows/vol_monitor.yml`: 30분 간격으로 실행하도록 설정하고 결과·저널·상태를 갱신합니다. GitHub Actions의 실제 실행 시각은 지연될 수 있습니다.

## 검증 상태와 한계

과거 옵션 전략 시뮬레이션은 호가와 시간가치 비용을 반영하려 했지만, 일부 꼬리 통계가 평가 시점 이후의 데이터로 적합됐고 옵션 가격에도 GARCH 대용치가 쓰였습니다. 따라서 그 결과는 독립적인 OOS 성과나 실거래 수익의 증거가 아닙니다. 손실과 해석상 문제가 확인된 `shock_propagator`, `variance_concentrated`, `causal_chain_monitor`는 명세서에 제거 이유를 남겼습니다. 전향 검증 기록과 페이퍼 주문 저널은 별도로 관리합니다.

Alpaca 주문 코드는 **페이퍼 실험용**으로 설계했습니다. 공개 코드만으로 배포 환경의 비공개 엔드포인트 설정이나 현재 체결 상태를 확인할 수 없으므로, 포털의 신호·백테스트·페이퍼 기록을 실계좌 운용 성과로 읽어서는 안 됩니다.

## 실행

Python 의존성은 `requirements.txt`와 `macro_research/requirements.txt`에 있습니다. 모니터 스냅샷은 `python vol_monitor_pipeline.py`로 만들 수 있습니다. 전체 분석과 주문 단계에는 외부 API 접근권한, 비공개 환경변수, 입력 캐시가 필요합니다. 포털은 `cd macro-portal && npm ci && npm run dev`로 실행합니다. 저장소의 생성 JSON은 화면 예시와 감사 기록이며, 재현용 원천 데이터 전체가 아닙니다.

**연구 원칙:** 기준을 먼저 정하고, 시간 순서를 지키며 검증하고, 실패한 규칙과 남은 편향을 기록합니다.


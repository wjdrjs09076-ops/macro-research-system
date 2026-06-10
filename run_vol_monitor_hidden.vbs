' Hidden wrapper — batch 를 창 없이(0) 비동기(False) 실행.
' Task Scheduler 에 batch 대신 이 vbs 를 등록하면 매시간 cmd 창이 안 뜬다.
' 로그는 그대로 vol_monitor.log 에 누적 — 사용자/AI 양쪽이 검증 가능.
Set WshShell = CreateObject("WScript.Shell")
WshShell.Run """C:\Users\wjdrj\Desktop\invest\run_vol_monitor.bat""", 0, False

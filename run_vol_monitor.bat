@echo off
REM Vol Monitor Hourly Pipeline

set LOGFILE=C:\Users\wjdrj\Desktop\invest\vol_monitor.log
set PYTHON=C:\Python314\python.exe
set VERCEL=C:\Users\wjdrj\AppData\Roaming\npm\vercel.cmd

REM Load User environment variables (needed when run from Task Scheduler)
for /f "tokens=2*" %%A in ('reg query "HKCU\Environment" /v EIA_API_KEY 2^>nul') do set EIA_API_KEY=%%B
for /f "tokens=2*" %%A in ('reg query "HKCU\Environment" /v FRED_API_KEY 2^>nul') do set FRED_API_KEY=%%B

echo [%date% %time%] Starting vol monitor pipeline >> %LOGFILE%

REM Run pipeline
cd /d C:\Users\wjdrj\Desktop\invest
%PYTHON% vol_monitor_pipeline.py >> %LOGFILE% 2>&1
if errorlevel 1 (
    echo [%date% %time%] ERROR: pipeline failed >> %LOGFILE%
    exit /b 1
)

REM Refresh analytical caches (sector_returns, garch_vol, macro_levels, capm_iv)
REM 라이브 운영용 — yfinance 의 최신 일별 종가를 ontology 가 보도록.
REM 이게 없으면 ontology / kinetic 이 stale 데이터로 결정하게 됨.
cd /d C:\Users\wjdrj\Desktop\invest\macro_research
%PYTHON% data_loader.py >> %LOGFILE% 2>&1
if errorlevel 1 (
    echo [%date% %time%] WARN: data_loader returned error >> %LOGFILE%
)
%PYTHON% garch_estimator.py >> %LOGFILE% 2>&1
if errorlevel 1 (
    echo [%date% %time%] WARN: garch_estimator returned error >> %LOGFILE%
)
%PYTHON% implied_vol.py >> %LOGFILE% 2>&1
if errorlevel 1 (
    echo [%date% %time%] WARN: implied_vol returned error >> %LOGFILE%
)

REM Kinetic layer: exit checks + new straddle entries (RTH-guarded, no-op off hours)
REM Trigger source = ontology inference signals (default). Falls back to vol via --source vol.
REM --minimal: 0.5%/signal × max 2 candidates per strategy (2026-06-08~).
REM   첫 실거래 검증 단계 — N>=10 거래 누적까지 minimal 유지. 누적되면 통상 모드로 전환.
cd /d C:\Users\wjdrj\Desktop\invest\macro_research
%PYTHON% kinetic_executor.py --auto --minimal >> %LOGFILE% 2>&1
if errorlevel 1 (
    echo [%date% %time%] WARN: kinetic executor returned error >> %LOGFILE%
)

REM Feedback loop: regenerate per-rule sizing multipliers from closed-trade attribution.
REM No-op until trades close (writes empty {}); harmless to run every cycle.
%PYTHON% feedback.py >> %LOGFILE% 2>&1
if errorlevel 1 (
    echo [%date% %time%] WARN: feedback returned error >> %LOGFILE%
)

REM Deploy to Vercel — --cwd 명시 + < NUL 로 stdin 차단 (인터랙티브 prompt hang 방지)
set VERCEL_CWD=C:\Users\wjdrj\Desktop\invest\macro-portal
cd /d %VERCEL_CWD%
echo [%date% %time%] vercel cwd=%CD% >> %LOGFILE%
call "%VERCEL%" deploy --cwd "%VERCEL_CWD%" --yes --prod < NUL >> %LOGFILE% 2>&1
if errorlevel 1 (
    echo [%date% %time%] ERROR: vercel deploy failed >> %LOGFILE%
) else (
    echo [%date% %time%] Deployed to Vercel >> %LOGFILE%
)

echo [%date% %time%] Done >> %LOGFILE%

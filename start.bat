@echo off
title Chatra - Powered by IngeniousAnalytics.com (Port 8085)
cd /d "%~dp0"

echo ====================================================================
echo   Starting Chatra — Powered by IngeniousAnalytics.com
echo   URL: http://localhost:8085
echo   Chat Route: http://localhost:8085/#/chat/961ff56d-2f0b-4c65-b130-e4d2135f8c75
echo ====================================================================

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" server.py --port 8085
) else (
    python server.py --port 8085
)
if errorlevel 1 pause

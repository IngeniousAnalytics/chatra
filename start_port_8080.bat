@echo off
title ChatUI Local - Advisor Voice & LLM Studio (Port 8080)
cd /d "%~dp0"

echo ====================================================================
echo   Starting ChatUI Local on Port 8080
echo   URL: http://localhost:8080
echo   Chat Route: http://localhost:8080/#/chat/961ff56d-2f0b-4c65-b130-e4d2135f8c75
echo ====================================================================

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" server.py --port 8080
) else (
    python server.py --port 8080
)
if errorlevel 1 pause

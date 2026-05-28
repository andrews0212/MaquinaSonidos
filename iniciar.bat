@echo off
title Monitor Bomba - IA Stream
echo.
echo  ==============================================
echo   MONITOR DE BOMBA - Deteccion de Anomalias
echo  ==============================================
echo.
cd /d "%~dp0"
venv\Scripts\uvicorn.exe app:app --host 0.0.0.0 --port 8000 --reload
pause

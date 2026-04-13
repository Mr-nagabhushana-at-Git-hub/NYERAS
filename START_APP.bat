@echo off
setlocal
cd /d "%~dp0"
title NYERAS Retail UI
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0START_APP.ps1"
set "EXITCODE=%ERRORLEVEL%"
echo.
if not "%EXITCODE%"=="0" echo START_APP failed with exit code %EXITCODE%.
pause >nul
exit /b %EXITCODE%

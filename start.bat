@echo off
chcp 65001 >nul
title Genesis WebUI - Launcher

echo ========================================
echo Genesis WebUI Launcher
echo ========================================
echo.

REM Step 1: Close old backend processes
echo [1/4] Closing old backend processes...
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*apps\\genesis_webui_integrated.py*' } | ForEach-Object { Write-Output ('    - Stopping backend PID ' + $_.ProcessId); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" 2>nul
echo.

REM Step 2: Close browser windows (optional, kills all browser instances accessing localhost:7860)
echo [2/4] Closing old browser tabs...
powershell -NoProfile -Command "Get-Process | Where-Object { $_.ProcessName -match 'chrome|msedge|firefox' } | ForEach-Object { $_.CloseMainWindow() | Out-Null }" 2>nul
echo.

REM Step 3: Wait for port to be released
echo [3/4] Waiting for port release...
timeout /t 3 /nobreak >nul
echo.

REM Step 4: Start new backend with visible console
echo [4/4] Starting new backend...
echo.
echo ========================================
echo Backend Console (Keep this window open)
echo ========================================
echo.

REM Start backend in current window (shows logs)
E:\liliyuanshangmie\Genesis-webui-modular-integration\python313\python.exe apps\genesis_webui_integrated.py

REM If backend exits with error
if errorlevel 1 (
    echo.
    echo ========================================
    echo [ERROR] Backend failed to start
    echo ========================================
    echo Please check the error message above
    pause
    exit /b 1
)

REM If backend exits normally (shouldn't happen unless manually stopped)
echo.
echo Backend stopped.
pause

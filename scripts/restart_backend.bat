@echo off
chcp 65001 >nul

echo ========================================
echo Genesis WebUI - 后端重启工具
echo ========================================

REM 记录脚本路径与项目根目录
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."

REM 切换到项目根目录
cd /d "%PROJECT_ROOT%"

echo [1/3] 查找并关闭旧的后端进程...
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*apps\\genesis_webui_integrated.py*' } | ForEach-Object { Write-Output ('    - 终止进程 ID ' + $_.ProcessId); Stop-Process -Id $_.ProcessId -Force }" >nul

echo [2/3] 载入 Python313 配置...
set "PY_CONFIG=%PROJECT_ROOT%\python_config.bat"
if not exist "%PY_CONFIG%" (
    echo    ❌ 未找到 python_config.bat ，请确认文件存在于项目根目录。
    pause
    exit /b 1
)

call "%PY_CONFIG%"
if errorlevel 1 (
    echo    ❌ Python313 配置失败，请检查 python_config.bat
    pause
    exit /b 1
)

echo [3/3] 启动新的后端...
start "GenesisBackend" "%PYTHON_EXE%" apps\genesis_webui_integrated.py

echo.
echo ✅ 后端已重启，可以刷新浏览器访问。
echo 如果需要查看日志，请切换到名为 "GenesisBackend" 的命令窗口。

pause

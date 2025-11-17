@echo off
echo ========================================
echo Flux Standalone UI Launcher
echo ========================================
echo.

cd /d "%~dp0.."

REM 加载 Python313 配置
call python_config.bat
if errorlevel 1 (
    echo Python313 配置失败！
    pause
    exit /b 1
)

echo 使用 Python313...
"%PYTHON_EXE%" --version
echo.

echo.
echo Checking dependencies...
"%PYTHON_EXE%" -c "import diffusers" 2>nul
if errorlevel 1 (
    echo.
    echo diffusers not found. Installing...
    "%PYTHON_EXE%" -m pip install diffusers transformers accelerate
)

echo.
echo Starting Flux Standalone UI...
echo.
"%PYTHON_EXE%" apps\sd_module\flux_gradio_standalone.py

pause

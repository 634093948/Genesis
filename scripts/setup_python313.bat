@echo off
REM ========================================
REM Python313 环境配置脚本
REM ========================================

cd /d "%~dp0.."

set PYTHON_EXE=%~dp0..\python313\python.exe
set PIP_EXE=%~dp0..\python313\Scripts\pip.exe

echo ========================================
echo Python313 环境配置
echo ========================================
echo.

REM 检查 Python313
echo [1/4] 检查 Python313...
if not exist "%PYTHON_EXE%" (
    echo 错误: Python313 未找到！
    echo 路径: %PYTHON_EXE%
    pause
    exit /b 1
)

"%PYTHON_EXE%" --version
echo ✓ Python313 已找到
echo.

REM 升级 pip
echo [2/4] 升级 pip...
"%PYTHON_EXE%" -m pip install --upgrade pip
echo.

REM 安装 Flux 依赖
echo [3/4] 安装 Flux 依赖...
if exist requirements_flux.txt (
    "%PYTHON_EXE%" -m pip install -r requirements_flux.txt
) else (
    echo 安装核心依赖...
    "%PYTHON_EXE%" -m pip install torch torchvision
    "%PYTHON_EXE%" -m pip install diffusers transformers accelerate
    "%PYTHON_EXE%" -m pip install gradio pillow numpy
)
echo.

REM 验证安装
echo [4/4] 验证安装...
"%PYTHON_EXE%" -c "import torch; print(f'PyTorch: {torch.__version__}')"
"%PYTHON_EXE%" -c "import diffusers; print(f'diffusers: {diffusers.__version__}')" 2>nul
if errorlevel 1 (
    echo ⚠ diffusers 未安装
) else (
    echo ✓ diffusers 已安装
)
"%PYTHON_EXE%" -c "import gradio; print(f'gradio: {gradio.__version__}')" 2>nul
if errorlevel 1 (
    echo ⚠ gradio 未安装
) else (
    echo ✓ gradio 已安装
)
echo.

echo ========================================
echo ✅ Python313 环境配置完成！
echo ========================================
echo.
echo 现在可以运行:
echo   scripts\start_flux_standalone.bat
echo.

pause

@echo off
REM Start Flux Text-to-Image UI
REM Author: eddy
REM Date: 2025-11-16

echo ========================================
echo Flux Text-to-Image UI Launcher
echo ========================================
echo.

cd /d "%~dp0.."

echo Current directory: %CD%
echo.

echo Starting Flux Gradio UI...
echo.

python apps\sd_module\flux_gradio_ui.py

pause

@echo off
chcp 65001 >nul
echo ======================================================================
echo 复制 Infinite Talk 所需的 Custom Nodes
echo ======================================================================
echo.

set SOURCE_DIR=E:\liliyuanshangmie\Fuxkcomfy_lris_kernel_gen2-4_speed_safe\FuxkComfy\custom_nodes
set TARGET_DIR=E:\liliyuanshangmie\Genesis-webui-modular-integration\custom_nodes\Comfyui

echo 源目录: %SOURCE_DIR%
echo 目标目录: %TARGET_DIR%
echo.

echo ======================================================================
echo 开始复制...
echo ======================================================================
echo.

REM 1. ComfyUI-KJNodes
echo [1/11] 复制 ComfyUI-KJNodes...
xcopy /E /I /Y "%SOURCE_DIR%\ComfyUI-KJNodes" "%TARGET_DIR%\ComfyUI-KJNodes"
echo.

REM 2. ComfyUI-TorchCompileSpeed
echo [2/11] 复制 ComfyUI-TorchCompileSpeed...
xcopy /E /I /Y "%SOURCE_DIR%\ComfyUI-TorchCompileSpeed" "%TARGET_DIR%\ComfyUI-TorchCompileSpeed"
echo.

REM 3. ComfyUI-UniversalBlockSwap
echo [3/11] 复制 ComfyUI-UniversalBlockSwap...
xcopy /E /I /Y "%SOURCE_DIR%\ComfyUI-UniversalBlockSwap" "%TARGET_DIR%\ComfyUI-UniversalBlockSwap"
echo.

REM 4. ComfyUI-VideoHelperSuite
echo [4/11] 复制 ComfyUI-VideoHelperSuite...
xcopy /E /I /Y "%SOURCE_DIR%\ComfyUI-VideoHelperSuite" "%TARGET_DIR%\ComfyUI-VideoHelperSuite"
echo.

REM 5. ComfyUI-WanVideoDecode-Standalone
echo [5/11] 复制 ComfyUI-WanVideoDecode-Standalone...
xcopy /E /I /Y "%SOURCE_DIR%\ComfyUI-WanVideoDecode-Standalone" "%TARGET_DIR%\ComfyUI-WanVideoDecode-Standalone"
echo.

REM 6. ComfyUI-WanVideoWrapper
echo [6/11] 复制 ComfyUI-WanVideoWrapper...
xcopy /E /I /Y "%SOURCE_DIR%\ComfyUI-WanVideoWrapper" "%TARGET_DIR%\ComfyUI-WanVideoWrapper"
echo.

REM 7. ComfyUI_essentials
echo [7/11] 复制 ComfyUI_essentials...
xcopy /E /I /Y "%SOURCE_DIR%\ComfyUI_essentials" "%TARGET_DIR%\ComfyUI_essentials"
echo.

REM 8. audio-separation-nodes-comfyui
echo [8/11] 复制 audio-separation-nodes-comfyui...
xcopy /E /I /Y "%SOURCE_DIR%\audio-separation-nodes-comfyui" "%TARGET_DIR%\audio-separation-nodes-comfyui"
echo.

REM 9. comfy-mtb
echo [9/11] 复制 comfy-mtb...
xcopy /E /I /Y "%SOURCE_DIR%\comfy-mtb" "%TARGET_DIR%\comfy-mtb"
echo.

REM 10. comfyui-easy-use
echo [10/11] 复制 comfyui-easy-use...
xcopy /E /I /Y "%SOURCE_DIR%\comfyui-easy-use" "%TARGET_DIR%\comfyui-easy-use"
echo.

REM 11. comfyui_tinyterranodes
echo [11/11] 复制 comfyui_tinyterranodes...
xcopy /E /I /Y "%SOURCE_DIR%\comfyui_tinyterranodes" "%TARGET_DIR%\comfyui_tinyterranodes"
echo.

echo ======================================================================
echo 复制完成!
echo ======================================================================
echo.
echo 已复制 11 个 custom_nodes 文件夹
echo.
pause

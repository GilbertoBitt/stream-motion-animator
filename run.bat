@echo off
REM Stream Motion Animator - Quick Start with Model Selection

echo ========================================
echo Stream Motion Animator
echo ========================================
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv .venv
    echo Then: .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

REM Model Selection Menu
echo Select Animation Model:
echo   1. Custom ONNX Model (Character-specific, 85%% quality)
echo   2. Mock Model (Basic transforms, 20%% quality)
echo   3. Auto-detect (Use custom if available)
echo.
set /p MODEL_CHOICE="Enter choice (1-3, default 1): "

if "%MODEL_CHOICE%"=="" set MODEL_CHOICE=1
if "%MODEL_CHOICE%"=="1" set MODEL_TYPE=custom_onnx
if "%MODEL_CHOICE%"=="2" set MODEL_TYPE=mock
if "%MODEL_CHOICE%"=="3" set MODEL_TYPE=auto

echo.
echo Starting application with Camera 1...
echo Model: %MODEL_TYPE%
echo.
echo Controls:
echo   Q - Quit
echo   1-9 - Switch character
echo   Left/Right Arrow - Previous/Next character
echo   R - Reload characters
echo   T - Toggle stats
echo.

.venv\Scripts\python.exe src\main.py --camera 1 --model %MODEL_TYPE%

echo.
echo Application closed.
pause


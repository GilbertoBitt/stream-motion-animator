@echo off
REM Preprocess All Characters - Caching Tool
REM Run this after adding new characters or videos

echo ======================================================================
echo CHARACTER PREPROCESSING TOOL
echo ======================================================================
echo.
echo This will pre-process all characters for maximum performance.
echo.
echo What it does:
echo   - Extracts frames from all videos
echo   - Processes all images
echo   - Creates optimized cache
echo   - Speeds up loading by 10-30x
echo.
echo Run this ONCE after adding/updating characters.
echo.
pause
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv .venv
    echo Then: .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

echo Starting preprocessing...
echo.

.venv\Scripts\python.exe tools\preprocess_all_characters.py

echo.
echo ======================================================================
echo.
pause


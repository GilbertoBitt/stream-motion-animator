@echo off
REM Automatic LivePortrait Model Downloader
REM This will download and install LivePortrait from Hugging Face

echo ======================================================================
echo LIVEPORTRAIT MODEL - AUTOMATIC INSTALLER
echo ======================================================================
echo.
echo This will:
echo   - Download LivePortrait model from Hugging Face (~2GB)
echo   - Extract and install all files
echo   - Install required dependencies
echo   - Test the installation
echo.
echo Time required: 5-15 minutes (depends on internet speed)
echo.
echo Make sure you have:
echo   - Good internet connection
echo   - At least 5GB free disk space
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

echo Starting automatic download and installation...
echo.

.venv\Scripts\python.exe download_liveportrait_auto.py

echo.
echo ======================================================================
echo.
echo If installation was successful, run: run.bat
echo.
pause


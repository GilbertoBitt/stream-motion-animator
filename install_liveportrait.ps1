# Automatic LivePortrait Model Installer (PowerShell)
# Downloads and installs LivePortrait from Hugging Face

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "LIVEPORTRAIT MODEL - AUTOMATIC INSTALLER" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This will:" -ForegroundColor Yellow
Write-Host "  - Download LivePortrait model from Hugging Face (~2GB)"
Write-Host "  - Extract and install all files"
Write-Host "  - Install required dependencies"
Write-Host "  - Test the installation"
Write-Host ""
Write-Host "Time required: 5-15 minutes (depends on internet speed)" -ForegroundColor Green
Write-Host ""
Write-Host "Make sure you have:"
Write-Host "  - Good internet connection"
Write-Host "  - At least 5GB free disk space"
Write-Host ""
Read-Host "Press Enter to continue"
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run: python -m venv .venv"
    Write-Host "Then: .venv\Scripts\pip install -r requirements.txt"
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Starting automatic download and installation..." -ForegroundColor Green
Write-Host ""

.\.venv\Scripts\python.exe download_liveportrait_auto.py

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "If installation was successful, run: run.bat" -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to exit"


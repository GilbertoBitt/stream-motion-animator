# Preprocess All Characters - Caching Tool (PowerShell)
# Run this after adding new characters or videos

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "CHARACTER PREPROCESSING TOOL" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This will pre-process all characters for maximum performance." -ForegroundColor Yellow
Write-Host ""
Write-Host "What it does:"
Write-Host "  - Extracts frames from all videos"
Write-Host "  - Processes all images"
Write-Host "  - Creates optimized cache"
Write-Host "  - Speeds up loading by 10-30x"
Write-Host ""
Write-Host "Run this ONCE after adding/updating characters." -ForegroundColor Green
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

Write-Host "Starting preprocessing..." -ForegroundColor Green
Write-Host ""

.\.venv\Scripts\python.exe tools\preprocess_all_characters.py

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to exit"


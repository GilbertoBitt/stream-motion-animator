# Stream Motion Animator - Quick Start (PowerShell)
# This script runs the application with optimal settings

Write-Host "========================================"  -ForegroundColor Cyan
Write-Host "Stream Motion Animator" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run: python -m venv .venv"
    Write-Host "Then: .venv\Scripts\pip install -r requirements.txt"
    Read-Host "Press Enter to exit"
    exit 1
}

# Run the application with Camera 1 (which works better than Camera 0)
Write-Host "Starting application with Camera 1..." -ForegroundColor Green
Write-Host ""
Write-Host "Controls:" -ForegroundColor Yellow
Write-Host "  Q - Quit"
Write-Host "  1-9 - Switch character"
Write-Host "  Left/Right Arrow - Previous/Next character"
Write-Host "  R - Reload characters"
Write-Host "  T - Toggle stats"
Write-Host ""

.\.venv\Scripts\python.exe src\main.py --camera 1

Write-Host ""
Write-Host "Application closed." -ForegroundColor Green
Read-Host "Press Enter to exit"


@echo off
REM Create Custom Character Model from Test Character
REM This processes your 32 character frames to create a custom ONNX model

echo ======================================================================
echo CUSTOM CHARACTER MODEL GENERATOR
echo ======================================================================
echo.
echo This will create a custom animation model from your Test character's
echo 32 expression frames for LivePortrait-quality animation.
echo.
echo What it does:
echo   - Processes all 32 frames in assets/characters/Test/
echo   - Extracts facial landmarks and features
echo   - Creates character-specific animation model
echo   - Enables expression matching (happy, sad, surprised, etc.)
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

echo Starting character model creation...
echo.

.venv\Scripts\python.exe tools\create_character_model.py

echo.
echo ======================================================================
echo.
if %ERRORLEVEL% EQU 0 (
    echo SUCCESS! Custom character model created.
    echo.
    echo Next step: Run the application with run.bat
    echo Your character will now animate with its 32 unique expressions!
) else (
    echo WARNING: Model creation may have had issues.
    echo Check the output above for details.
)
echo.
pause


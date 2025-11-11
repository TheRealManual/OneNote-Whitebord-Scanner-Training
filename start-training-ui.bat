@echo off
echo ============================================================
echo ML Training UI Launcher
echo ============================================================
echo.

cd /d "%~dp0local-ai-backend"

echo Checking Python environment...
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found in PATH
    echo Please install Python 3.9+ or activate your virtual environment
    pause
    exit /b 1
)

echo.
echo Starting Training UI server...
echo.
echo Open your browser to: http://localhost:5001
echo.
echo Press Ctrl+C to stop the server
echo ============================================================
echo.

python training_ui.py

pause

@echo off
REM Run all tests and verify no skips exist
REM Usage: scripts\run_tests.bat

echo ============================================================
echo Running pytest...
echo ============================================================
python -m pytest tests/ -q --maxfail=1
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo.
echo ============================================================
echo Verifying no skips/xfails...
echo ============================================================
python scripts/verify_no_skips.py
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo.
echo ============================================================
echo ALL TESTS PASSED
echo ============================================================

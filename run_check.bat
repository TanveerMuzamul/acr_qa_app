@echo off
setlocal
cd /d %~dp0

echo ==============================
echo FIXING PATH + RUNNING TESTS
echo ==============================

if not exist venv\Scripts\python.exe (
  echo Virtual environment not found. Run setup_local.bat first.
  pause
  exit /b 1
)

call venv\Scripts\activate.bat
set PYTHONPATH=%cd%

echo Running tests...
python -m pytest -v --maxfail=50
if errorlevel 1 (
  echo.
  echo Tests failed. App will not start.
  pause
  exit /b 1
)

echo ==============================
echo STARTING APP

echo ==============================
python run.py
pause

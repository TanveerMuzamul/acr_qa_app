@echo off
setlocal
cd /d %~dp0

echo ==============================
echo SETTING UP LOCAL ENVIRONMENT
echo ==============================

if exist venv\Scripts\python.exe goto use_existing

echo Creating virtual environment...
py -3 -m venv venv
if errorlevel 1 python -m venv venv
if errorlevel 1 (
  echo Failed to create virtual environment.
  pause
  exit /b 1
)

:use_existing
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if errorlevel 1 (
  echo Setup failed.
  pause
  exit /b 1
)

echo.
echo Setup complete.
echo Run tests with: run_check.bat
echo Start app with: python run.py
pause

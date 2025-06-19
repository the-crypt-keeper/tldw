@echo off
REM Startup script for tldw application on Windows

echo =========================================
echo Starting tldw Application
echo =========================================

REM Check Python
echo Checking Python version...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.8 or higher.
    pause
    exit /b 1
)
python --version

REM Check ffmpeg
echo Checking ffmpeg...
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: ffmpeg not found. Audio/video processing may not work.
    echo Download from: https://ffmpeg.org/download.html
)

REM Create directories
echo Creating directories...
if not exist "Databases" mkdir Databases
if not exist "Logs" mkdir Logs
if not exist "Config_Files" mkdir Config_Files
echo Directories created

REM Install dependencies
if exist "requirements.txt" (
    echo Installing/updating dependencies...
    python -m pip install --quiet --upgrade pip
    python -m pip install --quiet -r requirements.txt
    echo Dependencies installed
) else (
    echo ERROR: requirements.txt not found
)

REM Run health check
if exist "check_app_health.py" (
    echo Running health check...
    python check_app_health.py
)

REM Start application
echo.
echo Starting tldw GUI...
echo =========================================

if exist "app_fixed.py" (
    echo Using enhanced launcher...
    python app_fixed.py -gui %*
) else (
    echo Using standard launcher...
    python summarize.py -gui %*
)

pause
@echo off
echo 🪐 Exoplanet Explorer Launcher
echo ==============================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Check if required packages are installed
echo 📦 Checking dependencies...
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo ❌ Streamlit not found. Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
)

echo ✅ Dependencies ready
echo.

REM Start the application
echo 🚀 Starting Exoplanet Explorer...
python launcher.py

pause
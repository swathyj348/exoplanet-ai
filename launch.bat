@echo off
title Exoplanet Explorer Launcher
color 0A

echo.
echo  ===============================================
echo  🌌 Exoplanet Explorer - NASA ML Platform 🌌
echo  ===============================================
echo.
echo  Starting the integrated application...
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo ✅ Python detected
echo.

REM Install dependencies if requirements.txt exists
if exist requirements.txt (
    echo 📦 Installing/updating dependencies...
    pip install -r requirements.txt
    echo.
)

REM Check if models directory exists
if not exist models\xgb_k2_adapt.pkl (
    echo ⚠️  Model file not found at models\xgb_k2_adapt.pkl
    echo The application may not work correctly without the trained model
    echo.
)

echo 🚀 Launching Exoplanet Explorer...
echo.
echo 📱 Landing Page will be available at: http://localhost:5000
echo 🤖 Streamlit App will be available at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start the Flask server
python server.py

echo.
echo Server stopped. Press any key to exit...
pause >nul
@echo off
echo ================================================================
echo 🌌 EXOPLANET EXPLORER - FRONTEND + BACKEND LAUNCHER
echo ================================================================
echo.

cd /d "C:\Users\Swathy\OneDrive\Desktop\explnt final\explnt"

echo 🔧 Activating virtual environment...
call "..\explnt_env\Scripts\activate.bat"

echo 📋 Checking Flask installation...
python -c "import flask; print('Flask version:', flask.__version__)"

echo 🌐 Starting Flask server for frontend...
echo Landing page will be available at: http://localhost:5000
echo.
python server.py

pause
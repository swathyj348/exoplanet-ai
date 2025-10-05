@echo off
echo 🚀 Starting Exoplanet Explorer...
echo.

REM Change to the application directory
cd /d "e:\Exoplanet-new\exoplanet-explorer"

REM Check if Python is available
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Install required packages if they don't exist
echo 📦 Installing required packages...
pip install streamlit plotly pandas numpy scikit-learn shap xgboost flask flask-cors requests

echo.
echo 🌐 Starting landing page server (Flask)...
start "Exoplanet Landing Page" python server.py

timeout /t 3 /nobreak > nul

echo.
echo 🔬 Starting Streamlit ML application...
echo Visit http://localhost:5000 for the landing page
echo Visit http://localhost:8501 for direct Streamlit access
echo.
echo Press Ctrl+C to stop the servers
echo.

streamlit run app.py
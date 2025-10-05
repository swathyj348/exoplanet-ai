@echo off
echo ================================================================
echo 🌌 EXOPLANET EXPLORER - QUICK LAUNCH
echo ================================================================
echo.

cd /d "C:\Users\Swathy\OneDrive\Desktop\explnt final\explnt"

echo 🔧 Activating virtual environment...
call "..\explnt_env\Scripts\activate.bat"

echo 🚀 Starting integrated application...
python launch_integrated.py

pause
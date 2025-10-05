@echo off
echo 🌌 Starting Exoplanet Explorer...

cd /d "C:\Users\Swathy\OneDrive\Desktop\explnt final\explnt"
echo Current directory: %CD%

echo 🔧 Activating virtual environment...
call "..\explnt_env\Scripts\activate.bat"

echo 📋 Checking Python and packages...
python --version
python -c "import streamlit; print('Streamlit version:', streamlit.__version__)"

echo 🚀 Starting Streamlit application...
python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0

pause
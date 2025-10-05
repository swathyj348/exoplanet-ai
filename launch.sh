#!/bin/bash

# Exoplanet Explorer Launcher Script
# NASA-inspired ML platform for exoplanet detection

echo "==============================================="
echo "🌌 Exoplanet Explorer - NASA ML Platform 🌌"
echo "==============================================="
echo ""
echo "Starting the integrated application..."
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "❌ Python is not installed or not in PATH"
        echo "Please install Python 3.8+ and try again"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo "✅ Python detected"
echo ""

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "📦 Installing/updating dependencies..."
    $PYTHON_CMD -m pip install -r requirements.txt
    echo ""
fi

# Check if model file exists
if [ ! -f "models/xgb_k2_adapt.pkl" ]; then
    echo "⚠️  Model file not found at models/xgb_k2_adapt.pkl"
    echo "The application may not work correctly without the trained model"
    echo ""
fi

echo "🚀 Launching Exoplanet Explorer..."
echo ""
echo "📱 Landing Page will be available at: http://localhost:5000"
echo "🤖 Streamlit App will be available at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the Flask server
$PYTHON_CMD server.py
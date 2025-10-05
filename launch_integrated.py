#!/usr/bin/env python3
"""
Complete launcher for the Exoplanet Explorer project.
This script sets up and launches the full integrated experience.
"""

import os
import sys
import time
import subprocess
import threading
import webbrowser
from pathlib import Path
import requests

def check_port(port):
    """Check if a port is available."""
    try:
        response = requests.get(f'http://localhost:{port}', timeout=2)
        return True
    except requests.exceptions.RequestException:
        return False

def start_flask_server():
    """Start the Flask server for the landing page."""
    try:
        print("🚀 Starting Flask landing page server...")
        process = subprocess.Popen([
            sys.executable, 'server.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait for Flask to start
        for i in range(10):
            if check_port(5000):
                print("✅ Flask server is running on http://localhost:5000")
                return process
            time.sleep(1)
        
        print("❌ Flask server failed to start")
        return None
        
    except Exception as e:
        print(f"❌ Error starting Flask server: {e}")
        return None

def open_browser():
    """Open the landing page in the default browser."""
    time.sleep(3)  # Wait for servers to stabilize
    try:
        print("🌐 Opening browser...")
        webbrowser.open('http://localhost:5000')
    except Exception as e:
        print(f"⚠️ Could not open browser automatically: {e}")
        print("📱 Please open http://localhost:5000 manually")

def main():
    """Main launcher function."""
    print("=" * 60)
    print("🌌 EXOPLANET EXPLORER - INTEGRATED LAUNCH")
    print("=" * 60)
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    required_files = ['server.py', 'app.py', 'index.html', 'styles.css']
    missing_files = [f for f in required_files if not (current_dir / f).exists()]
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print(f"📁 Current directory: {current_dir}")
        print("💡 Please run this script from the explnt project directory")
        return
    
    print("✅ All required files found")
    print(f"📁 Working directory: {current_dir}")
    
    # Check if ports are available
    if check_port(5000):
        print("⚠️ Port 5000 is already in use")
        choice = input("Continue anyway? (y/n): ")
        if choice.lower() != 'y':
            return
    
    # Start Flask server
    flask_process = start_flask_server()
    if not flask_process:
        print("❌ Failed to start Flask server")
        return
    
    # Open browser in background
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    print("\n" + "=" * 60)
    print("🎉 EXOPLANET EXPLORER IS READY!")
    print("=" * 60)
    print("📱 Landing Page: http://localhost:5000")
    print("🌌 ML Application: Click 'Start Exploring' on the landing page")
    print("📚 Documentation: Check MODEL_FIX_ANALYSIS.md for details")
    print("🧪 Test Data: Use solar_system_test.csv for testing")
    print("\n💡 Press Ctrl+C to stop all services")
    print("=" * 60)
    
    try:
        # Keep the main process alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        if flask_process:
            flask_process.terminate()
        print("✅ All services stopped")

if __name__ == "__main__":
    main()
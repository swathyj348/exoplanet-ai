"""
Simple launcher script for the Exoplanet Explorer application.
This script starts both the landing page server and Streamlit application.
"""

import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path
import threading

def is_port_in_use(port):
    """Check if a port is already in use."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return False
        except socket.error:
            return True

def start_streamlit():
    """Start the Streamlit application."""
    try:
        print("🚀 Starting Streamlit application...")
        
        # Check if Streamlit is already running
        if is_port_in_use(8501):
            print("✅ Streamlit is already running on http://localhost:8501")
            return True
        
        # Start Streamlit
        cmd = [sys.executable, '-m', 'streamlit', 'run', 'app.py', '--server.port', '8501']
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for Streamlit to start
        time.sleep(3)
        
        if is_port_in_use(8501):
            print("✅ Streamlit started successfully on http://localhost:8501")
            return True
        else:
            print("❌ Failed to start Streamlit")
            return False
            
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")
        return False

def start_simple_server():
    """Start a simple HTTP server for the landing page."""
    try:
        import http.server
        import socketserver
        
        port = 3000
        
        # Check if port is already in use
        if is_port_in_use(port):
            print(f"✅ Server already running on http://localhost:{port}")
            return True
        
        # Change to the directory containing the HTML files
        os.chdir(Path(__file__).parent)
        
        print(f"🌐 Starting landing page server on http://localhost:{port}")
        
        # Start simple HTTP server in background process
        cmd = [sys.executable, '-m', 'http.server', str(port)]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for server to start
        time.sleep(2)
        
        # Check if server started successfully
        if is_port_in_use(port):
            print(f"✅ Landing page server started successfully")
            return True
        else:
            print(f"❌ Failed to start landing page server")
            return False
            
    except Exception as e:
        print(f"❌ Error starting landing page server: {e}")
        return False

def main():
    """Main launcher function."""
    print("🪐 Exoplanet Explorer Launcher")
    print("=" * 50)
    
    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check if required files exist
    required_files = ['index.html', 'styles.css', 'script.js', 'app.py']
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return
    
    print("📁 All required files found")
    
    # Start the landing page server
    if start_simple_server():
        time.sleep(1)
        
        # Open the landing page in the default browser
        print("🌐 Opening landing page in browser...")
        webbrowser.open('http://localhost:3000')
        
        print("\n" + "=" * 50)
        print("✨ Exoplanet Explorer is now running!")
        print("🌐 Landing Page: http://localhost:3000")
        print("🚀 Click 'Start Exploring' to launch the ML application")
        print("\n💡 Instructions:")
        print("1. Visit the landing page at http://localhost:3000")
        print("2. Click 'Start Exploring' button")
        print("3. Follow the instructions to run: streamlit run app.py")
        print("4. The ML application will be available at http://localhost:8501")
        print("\nPress Ctrl+C to stop all servers")
        print("=" * 50)
        
        try:
            # Keep the script running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Shutting down Exoplanet Explorer...")
            
    else:
        print("❌ Failed to start the application")
        print("\n📋 Manual Setup:")
        print("1. Open a terminal in this directory")
        print("2. Run: python -m http.server 3000")
        print("3. Visit http://localhost:3000 in your browser")
        print("4. Click 'Start Exploring' and follow instructions")

if __name__ == "__main__":
    main()
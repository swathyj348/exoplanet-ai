"""
Alternative simple launcher for the Exoplanet Explorer.
Uses subprocess to start HTTP server reliably on Windows.
"""

import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path

def main():
    """Main launcher function."""
    print("🪐 Exoplanet Explorer - NASA ML Research Platform")
    print("=" * 55)
    
    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check if required files exist
    required_files = ['index.html', 'styles.css', 'script.js', 'app.py']
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        input("Press Enter to exit...")
        return
    
    print("📁 All required files found")
    print("🌐 Starting landing page server...")
    
    try:
        # Start HTTP server for landing page
        cmd = [sys.executable, '-m', 'http.server', '3000']
        server_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Give server time to start
        time.sleep(3)
        
        # Check if process is still running (indicates success)
        if server_process.poll() is None:
            print("✅ Landing page server started successfully!")
            print("🌐 Opening browser...")
            
            # Open browser
            webbrowser.open('http://localhost:3000')
            
            print("\n" + "=" * 55)
            print("✨ Exoplanet Explorer is now running!")
            print()
            print("🌐 Landing Page: http://localhost:3000")
            print("🚀 Click 'Start Exploring' to launch the ML application")
            print()
            print("💡 Next Steps:")
            print("1. The landing page should open automatically in your browser")
            print("2. Click the 'Start Exploring' button")
            print("3. Follow the instructions to run the Streamlit app:")
            print("   streamlit run app.py")
            print("4. The ML application will be at http://localhost:8501")
            print()
            print("Press Ctrl+C to stop the server")
            print("=" * 55)
            
            try:
                # Wait for user to stop
                server_process.wait()
            except KeyboardInterrupt:
                print("\n👋 Shutting down server...")
                server_process.terminate()
                
        else:
            print("❌ Failed to start server")
            stdout, stderr = server_process.communicate()
            if stderr:
                print(f"Error: {stderr.decode()}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        
    print("\n📋 Manual Alternative:")
    print("If automatic start failed, you can manually run:")
    print("1. python -m http.server 3000")
    print("2. Open http://localhost:3000 in your browser")
    print("3. Click 'Start Exploring' and follow instructions")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
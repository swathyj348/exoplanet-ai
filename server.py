"""
Flask backend to handle Streamlit launching from the HTML landing page.
This provides a cleaner integration between the HTML frontend and Streamlit app.
"""

import os
import subprocess
import sys
import time
import threading
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variable to track Streamlit process
streamlit_process = None
streamlit_port = 8501

def is_streamlit_running():
    """Check if Streamlit is already running on the specified port."""
    try:
        response = requests.get(f'http://localhost:{streamlit_port}', timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def start_streamlit_in_background():
    """Start Streamlit in a background thread."""
    global streamlit_process
    
    try:
        # Change to the directory containing app.py
        app_dir = Path(__file__).parent
        os.chdir(app_dir)
        
        # Start Streamlit
        cmd = [sys.executable, '-m', 'streamlit', 'run', 'app.py', '--server.port', str(streamlit_port)]
        streamlit_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a moment for Streamlit to start
        time.sleep(3)
        return True
        
    except Exception as e:
        print(f"Error starting Streamlit: {e}")
        return False

@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory('.', 'index.html')

@app.route('/styles.css')
def styles():
    """Serve the CSS file."""
    return send_from_directory('.', 'styles.css')

@app.route('/script.js')
def script():
    """Serve the JavaScript file."""
    return send_from_directory('.', 'script.js')

@app.route('/start-streamlit', methods=['POST'])
def start_streamlit():
    """API endpoint to start Streamlit application."""
    try:
        # Check if Streamlit is already running
        if is_streamlit_running():
            return jsonify({
                'success': True,
                'message': 'Streamlit is already running',
                'url': f'http://localhost:{streamlit_port}'
            })
        
        # Start Streamlit in background thread
        def launch_streamlit():
            start_streamlit_in_background()
        
        thread = threading.Thread(target=launch_streamlit)
        thread.daemon = True
        thread.start()
        
        # Give Streamlit a moment to start
        time.sleep(2)
        
        # Check if it started successfully
        if is_streamlit_running():
            return jsonify({
                'success': True,
                'message': 'Streamlit started successfully',
                'url': f'http://localhost:{streamlit_port}'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to start Streamlit',
                'error': 'Streamlit did not start within the expected time'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': 'Error starting Streamlit',
            'error': str(e)
        }), 500

@app.route('/status')
def status():
    """Check the status of the Streamlit application."""
    running = is_streamlit_running()
    return jsonify({
        'streamlit_running': running,
        'streamlit_url': f'http://localhost:{streamlit_port}' if running else None
    })

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("🚀 Starting Exoplanet Explorer Landing Server...")
    print(f"📱 Landing Page: http://localhost:5000")
    print(f"🌌 Streamlit will be available at: http://localhost:{streamlit_port}")
    print("\n💡 Click 'Start Exploring' on the landing page to launch the ML application!")
    
    # Start Flask server
    app.run(host='0.0.0.0', port=5000, debug=False)
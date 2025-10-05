# Exoplanet Explorer - Integrated Landing Page + Streamlit App

This project combines a beautiful NASA-inspired landing page with a powerful Streamlit-based machine learning application for exoplanet detection and analysis.

## 🌟 Features

- **Clean, Professional Landing Page**: NASA-inspired design with responsive layout
- **Integrated Launch System**: Click "Start Exploring" to launch the ML application
- **Streamlit ML App**: Full-featured exoplanet classification with SHAP explainability
- **Seamless User Experience**: Smooth transition from landing page to application

## 🚀 Quick Start

### Option 1: Use the Launcher (Recommended)
1. Double-click `start_exoplanet_explorer.bat` (Windows)
2. Wait for both servers to start
3. Visit http://localhost:5000 for the landing page
4. Click "Start Exploring" to access the ML application

### Option 2: Manual Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the landing page server:
   ```bash
   python server.py
   ```

3. In a new terminal, start Streamlit:
   ```bash
   streamlit run app.py
   ```

4. Access the applications:
   - Landing Page: http://localhost:5000
   - Streamlit App: http://localhost:8501

## 🎨 Design Integration

The landing page is based on the beautiful NASA_Space repository design, featuring:

- **Animated Starfield Background**: Dynamic space-themed visuals
- **Responsive Design**: Works on all device sizes
- **NASA Color Palette**: Professional space agency aesthetic
- **Smooth Animations**: Engaging user interactions
- **Production-Ready CSS**: Clean, maintainable code

## 🔬 ML Application Features

- **Exoplanet Classification**: Predict Candidate, Confirmed, or False Positive
- **SHAP Explainability**: Understand model decisions
- **NASA Data Integration**: Uses real Kepler mission data
- **Interactive Visualizations**: Plotly-powered charts and graphs
- **Batch Processing**: Analyze multiple objects at once

## 📁 Project Structure

```
exoplanet-explorer/
├── index.html              # Landing page HTML
├── styles.css              # NASA-inspired CSS styles
├── script.js               # JavaScript for interactions
├── server.py               # Flask backend for integration
├── app.py                  # Main Streamlit application
├── start_exoplanet_explorer.bat  # Windows launcher
├── requirements.txt        # Python dependencies
├── models/                 # ML model files
├── data/                   # NASA datasets
└── README.md              # This file
```

## 🛠️ Technical Details

### Landing Page Stack
- **HTML5**: Semantic, accessible markup
- **CSS3**: Modern features (Grid, Flexbox, Custom Properties)
- **Vanilla JavaScript**: No frameworks, fast loading
- **Flask Backend**: RESTful API for Streamlit integration

### ML Application Stack
- **Streamlit**: Interactive web application framework
- **XGBoost**: High-performance gradient boosting
- **SHAP**: Model explainability and interpretability
- **Plotly**: Interactive visualizations
- **Pandas/NumPy**: Data processing

## 🌐 Browser Support

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## 🔧 Troubleshooting

### Streamlit Won't Start
```bash
# Check if port 8501 is in use
netstat -an | findstr :8501

# Kill process using the port (Windows)
taskkill /f /pid <PID>

# Restart with different port
streamlit run app.py --server.port 8502
```

### Landing Page Issues
- Ensure Flask dependencies are installed: `pip install flask flask-cors`
- Check that port 5000 is available
- Verify all HTML/CSS/JS files are in the same directory

### Model Files Missing
- Check that `models/` directory contains the ML model files
- Ensure `data/` directory has the NASA dataset files

## 📊 Usage Examples

1. **Quick Demo**: Use the sample K2 data for immediate results
2. **Custom Analysis**: Upload your own NASA-format CSV files
3. **Batch Processing**: Analyze multiple objects and download results
4. **Model Interpretation**: Use SHAP visualizations to understand predictions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes (maintaining the NASA design aesthetic)
4. Test the integration thoroughly
5. Submit a pull request

## 📄 License

This project uses NASA public data and follows open science principles.

## 🙏 Credits

- **Design Inspiration**: NASA_Space repository by Neelakandan-Nampoothiri
- **Data Source**: NASA Exoplanet Archive, Kepler Mission
- **ML Framework**: Streamlit and scikit-learn ecosystem

## 🌌 About

The Exoplanet Explorer combines cutting-edge machine learning with NASA's wealth of astronomical data to make exoplanet detection accessible to researchers, students, and space enthusiasts. The integrated landing page provides a professional entry point that reflects the serious scientific work happening behind the scenes.

---

*"Two possibilities exist: either we are alone in the Universe or we are not. Both are equally terrifying."* - Arthur C. Clarke
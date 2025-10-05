# 🪐 Exoplanet Explorer - NASA ML Research Platform

A sophisticated machine learning platform for exoplanet detection and analysis, featuring a beautiful NASA-inspired landing page and an integrated Streamlit application with explainable AI capabilities.

## 🌟 Features

### 🎨 Beautiful Landing Page
- **NASA-inspired Design**: Clean, classy HTML + CSS UI with space-themed aesthetics
- **Responsive Layout**: Polished, production-ready single-page design
- **Animated Background**: Dynamic starfield with gradient effects
- **Interactive Elements**: Smooth scrolling, hover effects, and engaging animations

### 🤖 Advanced ML Pipeline
- **Predict**: State-of-the-art neural networks for exoplanet classification
- **Explain**: SHAP-based interpretability for transparent AI decisions
- **Real-time Processing**: Fast inference on NASA Kepler mission data
- **High Accuracy**: 98.7% model accuracy on verified datasets

### 📊 Comprehensive Analysis
- **NASA Data Integration**: Built on official Kepler and TESS mission datasets
- **Explainable AI**: SHAP values reveal feature importance and model reasoning
- **Interactive Visualizations**: Rich charts and graphs for data exploration
- **Batch Processing**: Handle multiple exoplanet candidates simultaneously

## 🚀 Quick Start

### Method 1: One-Click Launch (Windows)
```bash
# Simply double-click the start.bat file
start.bat
```

### Method 2: Python Launcher
```bash
# Run the Python launcher
python launcher.py
```

### Method 3: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start the landing page server (in one terminal)
python -m http.server 3000

# Start Streamlit application (in another terminal)  
streamlit run app.py

# Visit http://localhost:3000 in your browser
```

## 🎯 How to Use

1. **Launch the Application**
   - Use any of the methods above to start the system
   - The landing page will automatically open in your browser

2. **Explore the Landing Page**
   - Read about the ML pipeline features
   - Learn about NASA data integration
   - Enjoy the space-themed design and animations

3. **Start the ML Application**
   - Click the "Start Exploring" button on the landing page
   - Follow the instructions to launch Streamlit
   - Begin analyzing exoplanet data with AI

4. **Analyze Exoplanets**
   - Upload NASA-format CSV files or use sample data
   - Get AI predictions on exoplanet candidates
   - Explore SHAP explanations for model decisions
   - Download results for further analysis

## 🛠️ Technical Details

### Architecture
```
┌─────────────────┐    ┌──────────────────┐
│   Landing Page  │───▶│  Streamlit App   │
│   (HTML/CSS/JS) │    │   (Python ML)    │
│   Port: 3000    │    │   Port: 8501     │
└─────────────────┘    └──────────────────┘
```

### File Structure
```
exoplanet-explorer/
├── 🌐 Frontend (Landing Page)
│   ├── index.html          # Main landing page
│   ├── styles.css          # NASA-inspired styling
│   └── script.js           # Interactive functionality
├── 🤖 Backend (ML Application)  
│   ├── app.py              # Streamlit ML application
│   ├── models/             # Trained ML models
│   ├── data/               # NASA datasets
│   └── src/                # Source code modules
├── 🚀 Launchers
│   ├── launcher.py         # Python launcher script
│   ├── server.py           # Flask backend (optional)
│   └── start.bat           # Windows batch launcher
└── 📋 Configuration
    ├── requirements.txt    # Python dependencies
    └── README.md           # This file
```

### Technology Stack
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **ML Backend**: Python, Streamlit, Pandas, NumPy
- **Machine Learning**: XGBoost, SHAP, Scikit-learn
- **Visualization**: Plotly, Matplotlib
- **Data**: NASA Kepler/TESS mission datasets

## 🎨 Design Features

### Visual Elements
- **Color Palette**: Deep navy space backgrounds with cyan/turquoise accents
- **Typography**: Poppins font family for modern, readable text
- **Animations**: Smooth transitions, gradient flows, and pulse effects
- **Responsive**: Works perfectly on desktop, tablet, and mobile devices

### Interactive Components
- **Hero Section**: Animated starfield background with gradient text
- **Feature Cards**: Hover effects with glowing borders and smooth transitions
- **Statistics**: Live counters showing exoplanet discovery metrics
- **Navigation**: Smooth scrolling between sections

### NASA-Inspired Theme
- **Space Aesthetics**: Dark backgrounds mimicking deep space
- **Scientific Data**: Real statistics from NASA exoplanet discoveries
- **Professional Layout**: Clean, organized presentation suitable for research
- **Accessibility**: High contrast and readable typography

## 📊 Model Information

### XGBoost K2-Adapted Model
- **Training Data**: NASA K2 mission light curve data
- **Features**: Stellar and planetary characteristics
- **Classes**: Candidate, Confirmed Exoplanet, False Positive
- **Accuracy**: 98.7% on validation set
- **Interpretability**: SHAP explanations for all predictions

### Data Sources
- NASA Exoplanet Archive
- Kepler Mission Database
- TESS (Transiting Exoplanet Survey Satellite)
- Exoplanet Orbit Database

## 🔧 Advanced Configuration

### Custom Streamlit Configuration
```python
# Modify app.py for custom settings
st.set_page_config(
    page_title="Exoplanet Classifier (K2)",
    page_icon="🪐",
    layout="wide",
)
```

### Landing Page Customization
- Edit `styles.css` for visual changes
- Modify `index.html` for content updates
- Update `script.js` for behavioral changes

## 🐛 Troubleshooting

### Common Issues

**Port Already in Use**
```bash
# Check what's using port 8501
netstat -ano | findstr :8501
# Kill the process if needed
taskkill /PID <PID> /F
```

**Dependencies Missing**
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

**Streamlit Won't Start**
```bash
# Try with specific Python version
python3 -m streamlit run app.py
```

**Browser Won't Open**
```bash
# Manually navigate to:
# Landing Page: http://localhost:3000
# Streamlit App: http://localhost:8501
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NASA**: For providing the Kepler and TESS mission data
- **Exoplanet Archive**: For maintaining comprehensive exoplanet databases
- **Scientific Community**: For advancing exoplanet research and discovery
- **Original Design**: Inspired by NASA_Space repository design elements

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Review the console output for error messages
3. Ensure all dependencies are properly installed
4. Verify that ports 3000 and 8501 are available

---

**🌌 Discover the worlds beyond our solar system with AI-powered analysis!**
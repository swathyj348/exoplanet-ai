# 🌌 Exoplanet Explorer - AI-Powered Detection Platform

A sophisticated machine learning platform for detecting and analyzing exoplanets using NASA's Kepler mission data. Features a beautiful, responsive landing page and powerful Streamlit-based ML application with explainable AI capabilities.

![Exoplanet Explorer](https://img.shields.io/badge/Status-Active-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red) ![License](https://img.shields.io/badge/License-MIT-yellow)

## 🚀 Features

### 🎨 Modern Landing Page
- **Clean, NASA-inspired design** with animated starfield background
- **Responsive layout** that works on all devices
- **Smooth animations** and interactive elements
- **Professional typography** using Poppins font
- **Gradient effects** and glassmorphism design elements

### 🤖 ML Application
- **XGBoost classifier** trained on NASA K2 mission data
- **98.7% accuracy** in exoplanet detection
- **SHAP explainability** for transparent AI decisions
- **Real-time predictions** with probability distributions
- **Interactive visualizations** using Plotly

### 📊 Key Capabilities
- Classify stellar objects as Candidates, Confirmed Exoplanets, or False Positives
- Process NASA CSV files with automatic feature engineering
- Generate SHAP waterfall charts for model interpretability
- Export prediction results with confidence scores
- Handle missing data with intelligent imputation

## 🛠 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/MilanGopakumar/explnt.git
   cd explnt
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the application**
   
   **Option A: Full integrated experience (Recommended)**
   ```bash
   python server.py
   ```
   Then visit `http://localhost:5000` for the landing page.
   
   **Option B: Direct Streamlit launch**
   ```bash
   streamlit run app.py
   ```
   Then visit `http://localhost:8501` for the ML app.

## 🎯 Usage Guide

### Landing Page Experience
1. Visit the landing page at `http://localhost:5000`
2. Explore the beautiful NASA-inspired interface
3. Read about the ML pipeline and NASA data sources
4. Click "Start Exploring" to launch the Streamlit application

### ML Application
1. **Data Input**: Choose sample NASA K2 data or upload your own CSV
2. **Predictions**: View classification results with confidence scores
3. **Explainability**: Analyze SHAP values to understand model decisions
4. **Visualization**: Explore probability distributions and feature importance
5. **Export**: Download results as CSV for further analysis

## 📁 Project Structure

```
exoplanet-explorer/
├── 📄 index.html          # Landing page HTML
├── 🎨 styles.css          # NASA-inspired CSS design
├── ⚡ script.js           # Interactive JavaScript
├── 🖥️ server.py           # Flask backend for integration
├── 🤖 app.py              # Main Streamlit application
├── 📋 requirements.txt    # Python dependencies
├── 🚀 launch.bat          # Windows launcher script
├── 📊 models/             # Trained ML models
│   └── xgb_k2_adapt.pkl   # XGBoost classifier
├── 📈 data/               # Sample NASA datasets
├── 🧪 tests/              # Test files
└── 📚 reports_k2_adapt_prod/ # Analysis reports
```

## 🔧 Technical Details

### Architecture
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Backend**: Flask (launcher), Streamlit (ML app)
- **ML Model**: XGBoost with SHAP explainability
- **Data**: NASA Exoplanet Archive, Kepler Mission data
- **Visualization**: Plotly.js for interactive charts

### Design System
- **Colors**: Deep space navy (#0b1230), cosmic indigo (#1a2847)
- **Accents**: Cyan/turquoise (#2dd4bf), warm amber (#ffb86b)
- **Typography**: Poppins font family
- **Animation**: Smooth transitions with CSS3
- **Responsive**: Mobile-first design approach

### ML Pipeline
1. **Data Preprocessing**: Feature engineering and normalization
2. **Model Training**: XGBoost on NASA K2 dataset
3. **Prediction**: Multi-class classification (3 categories)
4. **Explanation**: SHAP values for feature importance
5. **Validation**: Cross-validation and bias correction

## 🌟 Key Components

### Landing Page Features
- Animated starfield background with parallax scrolling
- Interactive hero section with statistics
- Feature cards with hover animations
- NASA data attribution and social links
- Responsive design for all screen sizes

### Streamlit Application
- Dark, NASA-inspired theme
- Interactive data upload and processing
- Real-time ML predictions with confidence scores
- SHAP explainability visualizations
- Downloadable results in CSV format

## 📊 Model Performance

- **Accuracy**: 98.7% on validation set
- **Classes**: Candidate, Confirmed Exoplanet, False Positive
- **Features**: 20+ astronomical parameters
- **Data Source**: NASA Exoplanet Archive K2 mission
- **Explainability**: SHAP values for transparency

## 🚀 Deployment Options

### Local Development
```bash
# Start the integrated system
python server.py

# Or run Streamlit directly
streamlit run app.py
```

### Production Deployment
- Deploy Flask server on cloud platforms (Heroku, AWS, GCP)
- Use process managers like Gunicorn for production
- Set up reverse proxy with Nginx for better performance
- Configure environment variables for different stages

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NASA Exoplanet Archive** for providing high-quality astronomical data
- **Kepler Mission** for revolutionary exoplanet discoveries
- **SHAP Library** for explainable AI capabilities
- **Streamlit Team** for the amazing ML app framework
- **Design inspiration** from NASA Space Apps and modern space exploration interfaces

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/MilanGopakumar/explnt/issues) page
2. Create a new issue with detailed description
3. Include system information and error logs

## 🌌 Future Enhancements

- [ ] Integration with TESS mission data
- [ ] Advanced deep learning models (CNNs, RNNs)
- [ ] Real-time data pipeline from NASA APIs
- [ ] Multi-language support
- [ ] Advanced visualization dashboard
- [ ] Mobile app development

---

**Built with ❤️ for space exploration and scientific discovery**

*Data courtesy of NASA Exoplanet Science Institute*

---

## Original Pipeline Documentation

Scripts
- src/preprocess.py: Build merged_tabular.csv from Kepler/K2 (handles NASA headers, maps labels).
- src/train_k2_adapt.py: Train optimized XGBoost adapted to K2; saves models/xgb_k2_adapt.pkl.
- src/evaluate.py: Evaluate any saved pipeline on a CSV; supports saved imputer/scaler/bias.

Quick start
1) Train (K2-adapted):
   - python src/train_k2_adapt.py
2) Evaluate on K2 PANDC:
   - python src/evaluate.py --model models/xgb_k2_adapt.pkl --csv data/k2pandc_2025.10.04_07.36.56.csv --out reports

Notes
- The saved pipeline includes a median SimpleImputer, StandardScaler, and a small per-class probability bias tuned on validation. You can override bias with --bias.
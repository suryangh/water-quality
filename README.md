# Water Quality Detection System 🌊

A machine learning-powered web application for detecting water quality using multiple ML algorithms. This system provides real-time water quality analysis through an intuitive web interface.

## ✨ Features

- **Multiple ML Models**: Random Forest, Neural Network, Decision Tree, KNN, and XGBoost
- **Web Interface**: User-friendly frontend for easy interaction
- **Real-time Prediction**: Instant water quality assessment
- **Model Comparison**: Compare different algorithms performance
- **Data Upload**: Support for CSV and Excel file uploads
- **Dockerized**: Easy deployment with Docker containers

## 🏗️ Architecture

```
├── backend/
│   ├── app.py                 # Flask API server
│   ├── ml_model/             # Pre-trained ML models
│   └── requirements.txt      # Python dependencies
├── frontend/
│   ├── index.html           # Main interface
│   ├── input.html           # Data input page
│   ├── model.html           # Model comparison
│   ├── result.html          # Results display
│   └── assets/              # Static files
└── Dockerfile               # Multi-stage Docker build
```

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/suryangh/water-quality.git
cd water-quality

# Build Docker image
docker build -t water-quality .

# Run the application
docker run -d --name water-quality -p 5000:5000 -p 5500:5500 water-quality
```

### Option 2: Local Development

```bash
# Backend setup
cd backend
python -m venv dasildat
source dasildat/bin/activate  # On Windows: dasildat\Scripts\activate
pip install -r requirements.txt
python app.py

# Frontend setup (in another terminal)
cd frontend
python serve_frontend.py
```

## 🌐 Access the Application

- **Frontend**: http://localhost:5500
- **Backend API**: http://localhost:5000

## 📊 Water Quality Parameters

The system analyzes the following water quality parameters:

| Parameter  | Safe Range | Unit  |
| ---------- | ---------- | ----- |
| Aluminium  | 0 - 0.2    | mg/L  |
| Ammonia    | 0 - 0.5    | mg/L  |
| Arsenic    | 0 - 0.01   | mg/L  |
| Barium     | 0 - 2.0    | mg/L  |
| Cadmium    | 0 - 0.005  | mg/L  |
| Chloramine | 0 - 4.0    | mg/L  |
| Chromium   | 0 - 0.1    | mg/L  |
| Copper     | 0 - 1.3    | mg/L  |
| Fluoride   | 0 - 4.0    | mg/L  |
| Bacteria   | 0          | count |
| Viruses    | 0          | count |

## 🤖 Machine Learning Models

### Available Models:

1. **Random Forest** - General purpose, robust predictions
2. **Neural Network** - Complex pattern recognition
3. **Decision Tree** - Simple, interpretable rules
4. **K-Nearest Neighbors (KNN)** - Pattern-based classification
5. **XGBoost** - High-performance gradient boosting

### Model Performance:

- All models are pre-trained on water quality datasets
- Real-time inference with optimized prediction times
- Model comparison features available in the web interface

## 🛠️ API Documentation

### Predict Water Quality

```http
POST /predict
Content-Type: application/json

{
  "model": "random_forest",
  "features": {
    "Aluminium": 0.1,
    "Ammonia": 0.2,
    "Arsenic": 0.005,
    // ... other parameters
  }
}
```

### Response

```json
{
  "prediction": "Safe",
  "confidence": 0.95,
  "model_used": "random_forest",
  "analysis": {
    "safe_parameters": ["Aluminium", "Ammonia"],
    "unsafe_parameters": ["Arsenic"]
  }
}
```

## 📁 File Upload

The system supports uploading water quality data in:

- CSV format
- Excel (.xlsx, .xls) format

Upload files through the web interface for batch predictions.

## 🐳 Docker Information

**Image Size**: ~560MB (optimized with multi-stage build)

**Container Specifications**:

- Base: Python 3.12 Alpine Linux
- Multi-stage build for minimal size
- Non-root user for security
- Exposed ports: 5000 (API), 5500 (Frontend)

## 🔧 Development

### Prerequisites

- Python 3.12+
- Docker (optional)
- Git

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/suryangh/water-quality.git
cd water-quality

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Run in development mode
cd backend
python app.py
```

## 📝 Environment Variables

```bash
FLASK_ENV=production          # Flask environment
PYTHONUNBUFFERED=1           # Python output buffering
PYTHONDONTWRITEBYTECODE=1    # Disable .pyc files
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

- **Team DASILDAT** -

## 📞 Support

If you have any questions or issues, please:

1. Check the [Issues](https://github.com/suryangh/water-quality/issues) page
2. Create a new issue if your problem isn't already reported
3. Provide detailed information about your setup and the issue

---

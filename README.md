# Wellora AI Healthcare Platform

A comprehensive AI-powered healthcare risk assessment platform that combines machine learning models with clinical knowledge to provide accurate disease risk predictions and generate detailed medical reports.

## 🚀 Features

- **Multi-Disease Risk Assessment**: Supports Diabetes, Heart Disease, Kidney Disease, and Liver Disease
- **Machine Learning Models**: Uses Logistic Regression, XGBoost, and LightGBM for predictions
- **Hybrid AI System**: Combines ML predictions with clinical rules for enhanced accuracy
- **AI-Generated Reports**: Powered by Google Gemini AI for comprehensive medical reports
- **Real-time Dashboard**: Live analytics and patient monitoring
- **Professional UI**: Modern, responsive design with Meditex template

## 🏥 Supported Conditions

1. **Diabetes Risk Assessment**
   - HbA1c levels, blood glucose, BMI analysis
   - Hypertension and heart disease history
   - Smoking status and lifestyle factors

2. **Heart Disease Risk Assessment**
   - Chest pain analysis, blood pressure monitoring
   - Cholesterol levels, ECG results
   - Exercise capacity and stress testing

3. **Kidney Disease Risk Assessment**
   - Blood urea, serum creatinine levels
   - Hemoglobin, blood cell counts
   - Hypertension and diabetes history

4. **Liver Disease Risk Assessment**
   - Bilirubin levels, liver enzymes
   - Protein levels, albumin analysis
   - Comprehensive liver function evaluation

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **ML Models**: Scikit-learn, XGBoost, LightGBM
- **AI Reports**: Google Gemini AI
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap
- **Data Processing**: Pandas, NumPy
- **Deployment**: Render (Cloud Platform)

## 📋 Prerequisites

- Python 3.11+
- pip (Python package installer)
- Git

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/joencrypts/AI-Disha.git
   cd AI-Disha
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   - Open your browser and go to `http://localhost:5000`

### Render Deployment

1. **Fork this repository** to your GitHub account

2. **Connect to Render**
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New +" and select "Web Service"
   - Connect your GitHub repository

3. **Configure deployment**
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Python Version**: 3.11.0

4. **Set environment variables**
   - `GEMINI_API_KEY`: Your Google Gemini API key
   - `PYTHON_VERSION`: 3.11.0

5. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment to complete
   - Access your live application

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
FLASK_ENV=production
```

### Model Training

To train new models with your data:

```bash
python train_models.py
```

## 📊 API Endpoints

- `GET /` - Homepage
- `GET /risk-assessment` - Risk assessment form
- `POST /assess_risk` - Process risk assessment
- `POST /generate_report` - Generate AI medical report
- `GET /doctor-dashboard` - Doctor dashboard
- `GET /api/dashboard-data` - Dashboard data API
- `GET /health` - Health check endpoint

## 🧪 Testing

Test the risk assessment with sample data:

```bash
python -c "
import requests
data = {
    'condition': 'diabetes',
    'age': 65,
    'gender': 1,
    'bmi': 35.0,
    'hypertension': 1,
    'heart_disease': 1,
    'smoking_history': 'current',
    'HbA1c_level': 8.5,
    'blood_glucose_level': 200
}
response = requests.post('http://localhost:5000/assess_risk', data=data)
print(response.json())
"
```

## 📁 Project Structure

```
AI-Disha/
├── app.py                 # Main Flask application
├── main.py               # AI Healthcare System
├── ml_models.py          # Machine Learning models
├── hybrid_inference.py    # Hybrid AI system
├── knowledge_base.py     # Clinical rules engine
├── dataset_processor.py  # Data preprocessing
├── feature_engineering.py # Feature creation
├── evaluation.py         # Model evaluation
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── render.yaml           # Render deployment config
├── .gitignore           # Git ignore rules
├── README.md            # This file
├── data/                # Training datasets
│   ├── diabetes_prediction_dataset.csv
│   ├── Heart_Disease_Prediction.csv
│   ├── Chronic_Kidney_Dsease_data.csv
│   └── Liver Patient Dataset (LPD)_train.csv
├── models/              # Trained ML models
├── results/             # Evaluation results
├── static/              # Static assets
└── templates/           # HTML templates
    ├── base.html
    ├── index.html
    ├── risk_assessment.html
    ├── doctor_dashboard.html
    └── ...
```

## 🔒 Security Features

- Input validation and sanitization
- Secure API endpoints
- Environment variable protection
- Error handling and logging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support, email support@wellora.com or create an issue in this repository.

## 🙏 Acknowledgments

- Google Gemini AI for report generation
- Meditex template for UI design
- Scikit-learn, XGBoost, and LightGBM communities
- Flask and Python communities

---

**Wellora AI Healthcare Platform** - Empowering healthcare with AI-driven insights.
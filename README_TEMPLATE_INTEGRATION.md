# AI-Powered Proactive Patient Risk Advisor
## With Meditex Template Integration

A comprehensive healthcare risk assessment system that combines advanced AI/ML algorithms with a professional medical website template for an enhanced user experience.

## 🎨 Template Integration

This project now integrates the **Meditex** medical template, providing:
- Professional medical website design
- Responsive layout for all devices
- Modern UI/UX components
- Medical-themed styling and icons
- Complete navigation system

## 🚀 Quick Start

### Option 1: Using the Startup Script (Recommended)
```bash
python start_app.py
```

### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## 🌐 Access Points

- **Main Application**: http://localhost:5000
- **Clinician Dashboard**: http://localhost:8050 (if enabled)

## 📁 Project Structure

```
ai-project/
├── app.py                          # Main Flask application with template integration
├── main.py                         # Core AI healthcare system
├── config.py                       # Configuration settings
├── start_app.py                    # Startup script
├── requirements.txt                # Python dependencies
├── templates/                      # HTML templates (Meditex-based)
│   ├── base.html                   # Base template with navigation
│   ├── index.html                  # Homepage
│   ├── risk_assessment.html        # Risk assessment interface
│   ├── about.html                  # About page
│   ├── contact.html                # Contact page
│   ├── departments.html            # Departments listing
│   ├── department.html             # Individual department pages
│   ├── faq.html                    # FAQ page
│   ├── voice_assistant.html        # Voice assistant interface
│   ├── 404.html                    # Error pages
│   ├── 500.html
│   └── search_results.html
├── static/                         # Static assets (Meditex template)
│   ├── assets/
│   │   ├── css/                    # Stylesheets
│   │   ├── js/                     # JavaScript files
│   │   ├── image/                  # Images and icons
│   │   └── fonts/                  # Font files
├── data/                           # Medical datasets
├── models/                         # Trained ML models
├── results/                        # Assessment results
└── template/                       # Original Meditex template
    └── meditex-final/
```

## 🎯 Key Features

### AI Healthcare System
- **Multi-Condition Risk Assessment**: Kidney, Diabetes, Heart, Liver diseases
- **Machine Learning Models**: XGBoost, LightGBM, Logistic Regression
- **Clinical Knowledge Base**: Evidence-based rules and guidelines
- **Hybrid Inference**: Combines ML predictions with clinical rules
- **Bulk Processing**: Assess multiple patients simultaneously
- **Voice Assistant**: Natural language interaction

### Template Features
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Professional Medical Theme**: Clean, modern medical website design
- **Interactive Components**: Carousels, accordions, forms
- **Navigation System**: Complete menu structure
- **Contact Forms**: Integrated contact and assessment forms
- **Error Handling**: Custom 404/500 pages

## 🔧 Configuration

### Environment Setup
1. Ensure Python 3.7+ is installed
2. Install required packages: `pip install -r requirements.txt`
3. Place medical datasets in the `data/` directory
4. Run the startup script: `python start_app.py`

### Template Customization
- **Colors**: Modify CSS variables in `static/assets/css/style.css`
- **Content**: Update text in HTML templates
- **Images**: Replace images in `static/assets/image/`
- **Navigation**: Modify menu items in `templates/base.html`

## 📱 Pages and Features

### Homepage (`/`)
- Hero slider with medical imagery
- Quick assessment form
- Department overview
- AI features showcase
- Testimonials and statistics

### Risk Assessment (`/risk-assessment`)
- Manual data entry form
- JSON input for complex data
- Bulk patient processing
- Sample patient scenarios
- Real-time results display

### Departments (`/departments`)
- Overview of all medical departments
- Individual department pages
- AI technology showcase
- Department statistics

### About (`/about`)
- Company information
- AI technology details
- Team showcase
- Statistics and achievements

### Contact (`/contact`)
- Contact form with validation
- Interactive map
- Contact information
- FAQ section

### Voice Assistant (`/voice-assistant`)
- Speech recognition interface
- Natural language processing
- Voice command examples
- Conversation history

## 🛠️ Technical Details

### Backend
- **Flask**: Web framework
- **Pandas/NumPy**: Data processing
- **Scikit-learn**: Machine learning
- **XGBoost/LightGBM**: Advanced ML models
- **Joblib**: Model persistence

### Frontend
- **HTML5/CSS3**: Modern web standards
- **Bootstrap**: Responsive framework
- **jQuery**: JavaScript library
- **Owl Carousel**: Image sliders
- **Font Awesome**: Icons

### AI/ML Pipeline
1. **Data Preprocessing**: Clean and normalize medical data
2. **Feature Engineering**: Create relevant features
3. **Model Training**: Train multiple ML algorithms
4. **Knowledge Base**: Clinical rules and guidelines
5. **Hybrid Inference**: Combine ML and clinical knowledge
6. **Risk Assessment**: Generate risk scores and recommendations

## 📊 Supported Medical Conditions

1. **Chronic Kidney Disease**
   - eGFR analysis
   - Proteinuria detection
   - Risk stratification

2. **Diabetes**
   - HbA1c monitoring
   - Glucose prediction
   - Complication risk

3. **Heart Disease**
   - ECG analysis
   - Cholesterol assessment
   - Chest pain evaluation

4. **Liver Disease**
   - Liver enzyme analysis
   - Bilirubin assessment
   - Cirrhosis risk

## 🔒 Security & Privacy

- **Data Encryption**: All patient data is encrypted
- **HIPAA Compliance**: Healthcare data protection standards
- **Secure Transmission**: HTTPS for all communications
- **No Data Storage**: Patient data is not permanently stored
- **Access Control**: Secure API endpoints

## 🚀 Deployment

### Local Development
```bash
python start_app.py
```

### Production Deployment
1. Set up a web server (Nginx/Apache)
2. Use WSGI server (Gunicorn/uWSGI)
3. Configure SSL certificates
4. Set up database for persistent storage
5. Configure environment variables

### Docker Deployment
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: Check this README and code comments
- **Issues**: Report bugs via GitHub issues
- **Contact**: Use the contact form on the website
- **Email**: contact@aihealthcare.com

## 🔄 Updates

### Version 2.0 - Template Integration
- ✅ Integrated Meditex medical template
- ✅ Responsive design implementation
- ✅ Enhanced user interface
- ✅ Professional medical styling
- ✅ Complete navigation system
- ✅ Contact forms and error pages
- ✅ Voice assistant interface

### Future Updates
- 🔄 Real-time notifications
- 🔄 Advanced analytics dashboard
- 🔄 Mobile app integration
- 🔄 API documentation
- 🔄 Multi-language support

## 📞 Contact

- **Website**: http://localhost:5000
- **Email**: contact@aihealthcare.com
- **Phone**: (555) 123-4567
- **Address**: AI Healthcare Technology Center, Medical Innovation Hub, NY 10001

---

**Built with ❤️ for better healthcare through AI**

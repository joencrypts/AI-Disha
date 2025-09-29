"""
Simplified Wellora AI Healthcare Platform - Deployment Version
This version focuses on core functionality without heavy ML dependencies
"""

import sys
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Production configuration
if os.getenv('FLASK_ENV') == 'production':
    app.config['DEBUG'] = False
else:
    app.config['DEBUG'] = True

# Initialize assessment storage
assessment_storage = {
    'assessments': [],
    'total_count': 0,
    'risk_counts': {'HIGH': 0, 'MODERATE': 0, 'LOW': 0},
    'condition_counts': {'kidney': 0, 'diabetes': 0, 'heart': 0, 'liver': 0}
}

# Configure Gemini AI (optional)
try:
    import google.generativeai as genai
    api_key = os.getenv('GEMINI_API_KEY', 'AIzaSyCCy99Oh-TrbsYlBlW0HGfesESMB7eS50k')
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    print("Gemini AI configured successfully!")
except Exception as e:
    print(f"Warning: Gemini AI not configured: {e}")
    gemini_model = None

@app.route('/')
def index():
    """Main homepage"""
    return render_template('index.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/contact')
def contact():
    """Contact page"""
    return render_template('contact.html')

@app.route('/departments')
def departments():
    """Departments page"""
    return render_template('departments.html')

@app.route('/risk-assessment')
def risk_assessment():
    """Risk assessment page"""
    return render_template('risk_assessment.html')

@app.route('/doctor-dashboard')
def doctor_dashboard():
    """Doctor dashboard"""
    return render_template('doctor_dashboard.html')

@app.route('/faq')
def faq():
    """FAQ page"""
    return render_template('faq.html')

@app.route('/assess_risk', methods=['POST'])
def assess_risk():
    """Simplified risk assessment endpoint"""
    try:
        # Get form data
        condition = request.form.get('condition', 'diabetes')  # Default to diabetes
        patient_data = {}
        
        print(f"Received condition: {condition}")
        print(f"Form data: {dict(request.form)}")
        
        # Collect all form data
        for key, value in request.form.items():
            if key != 'condition' and value:
                try:
                    # Try to convert to number if possible
                    if '.' in value:
                        patient_data[key] = float(value)
                    else:
                        patient_data[key] = int(value)
                except ValueError:
                    patient_data[key] = value
        
        print(f"Patient data: {patient_data}")
        
        # Simple risk assessment logic
        risk_score = 0
        risk_factors = []
        
        # Handle different conditions
        if condition == 'diabetes':
            if patient_data.get('age', 0) > 60:
                risk_score += 20
                risk_factors.append("Advanced age")
            if patient_data.get('bmi', 0) > 30:
                risk_score += 25
                risk_factors.append("High BMI")
            if patient_data.get('hypertension', 0) == 1:
                risk_score += 15
                risk_factors.append("Hypertension")
            if patient_data.get('heart_disease', 0) == 1:
                risk_score += 20
                risk_factors.append("Heart disease history")
            if patient_data.get('HbA1c_level', 0) > 6.5:
                risk_score += 30
                risk_factors.append("Elevated HbA1c")
        elif condition == 'heart':
            if patient_data.get('age', 0) > 65:
                risk_score += 20
                risk_factors.append("Advanced age")
            if patient_data.get('chest_pain_type', 0) > 2:
                risk_score += 25
                risk_factors.append("Chest pain symptoms")
            if patient_data.get('cholesterol', 0) > 240:
                risk_score += 20
                risk_factors.append("High cholesterol")
        elif condition == 'kidney':
            if patient_data.get('age', 0) > 60:
                risk_score += 15
                risk_factors.append("Advanced age")
            if patient_data.get('bp', 0) > 140:
                risk_score += 25
                risk_factors.append("High blood pressure")
            if patient_data.get('sc', 0) > 1.4:
                risk_score += 30
                risk_factors.append("Elevated serum creatinine")
        elif condition == 'liver':
            if patient_data.get('age', 0) > 50:
                risk_score += 15
                risk_factors.append("Advanced age")
            if patient_data.get('total_bilirubin', 0) > 1.2:
                risk_score += 25
                risk_factors.append("Elevated bilirubin")
            if patient_data.get('alt', 0) > 40:
                risk_score += 20
                risk_factors.append("Elevated ALT")
        
        # Ensure minimum risk score
        if risk_score == 0:
            risk_score = 10  # Minimum risk
            risk_factors.append("Baseline assessment")
        
        # Determine risk level
        if risk_score >= 70:
            risk_level = 'HIGH'
        elif risk_score >= 40:
            risk_level = 'MODERATE'
        else:
            risk_level = 'LOW'
        
        # Create result
        result = {
            'risk_level': risk_level,
            'risk_probability': min(risk_score / 100, 0.95),
            'confidence_score': 0.8,
            'risk_factors': risk_factors,
            'explanation': f"Based on the assessment, the patient shows {risk_level.lower()} risk for {condition}. Key factors: {', '.join(risk_factors) if risk_factors else 'No significant risk factors identified'}."
        }
        
        print(f"Assessment result: {result}")
        
        # Store assessment
        store_assessment_result(condition, result)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in assess_risk: {e}")
        return jsonify({'error': str(e)}), 500

def store_assessment_result(condition, result):
    """Store assessment result for dashboard"""
    import datetime
    
    assessment = {
        'id': f'P{assessment_storage["total_count"] + 1:03d}',
        'condition': condition,
        'risk_level': result['risk_level'],
        'risk_probability': result['risk_probability'],
        'confidence_score': result['confidence_score'],
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    assessment_storage['assessments'].append(assessment)
    assessment_storage['total_count'] += 1
    
    # Update counters
    risk_level = result['risk_level']
    if risk_level in assessment_storage['risk_counts']:
        assessment_storage['risk_counts'][risk_level] += 1
    
    if condition in assessment_storage['condition_counts']:
        assessment_storage['condition_counts'][condition] += 1
    
    # Keep only last 100 assessments
    if len(assessment_storage['assessments']) > 100:
        assessment_storage['assessments'] = assessment_storage['assessments'][-100:]

@app.route('/generate_report', methods=['POST'])
def generate_report():
    """Generate AI report using Gemini or fallback"""
    try:
        # Get form data
        condition = request.form.get('condition')
        patient_data = {}
        
        # Collect all form data
        for key, value in request.form.items():
            if key != 'condition' and value:
                try:
                    if '.' in value:
                        patient_data[key] = float(value)
                    else:
                        patient_data[key] = int(value)
                except ValueError:
                    patient_data[key] = value
        
        # Perform assessment first (simplified version)
        risk_score = 0
        risk_factors = []
        
        # Handle different conditions
        if condition == 'diabetes':
            if patient_data.get('age', 0) > 60:
                risk_score += 20
                risk_factors.append("Advanced age")
            if patient_data.get('bmi', 0) > 30:
                risk_score += 25
                risk_factors.append("High BMI")
            if patient_data.get('hypertension', 0) == 1:
                risk_score += 15
                risk_factors.append("Hypertension")
            if patient_data.get('heart_disease', 0) == 1:
                risk_score += 20
                risk_factors.append("Heart disease history")
            if patient_data.get('HbA1c_level', 0) > 6.5:
                risk_score += 30
                risk_factors.append("Elevated HbA1c")
        elif condition == 'heart':
            if patient_data.get('age', 0) > 65:
                risk_score += 20
                risk_factors.append("Advanced age")
            if patient_data.get('chest_pain_type', 0) > 2:
                risk_score += 25
                risk_factors.append("Chest pain symptoms")
            if patient_data.get('cholesterol', 0) > 240:
                risk_score += 20
                risk_factors.append("High cholesterol")
        elif condition == 'kidney':
            if patient_data.get('age', 0) > 60:
                risk_score += 15
                risk_factors.append("Advanced age")
            if patient_data.get('bp', 0) > 140:
                risk_score += 25
                risk_factors.append("High blood pressure")
            if patient_data.get('sc', 0) > 1.4:
                risk_score += 30
                risk_factors.append("Elevated serum creatinine")
        elif condition == 'liver':
            if patient_data.get('age', 0) > 50:
                risk_score += 15
                risk_factors.append("Advanced age")
            if patient_data.get('total_bilirubin', 0) > 1.2:
                risk_score += 25
                risk_factors.append("Elevated bilirubin")
            if patient_data.get('alt', 0) > 40:
                risk_score += 20
                risk_factors.append("Elevated ALT")
        
        # Ensure minimum risk score
        if risk_score == 0:
            risk_score = 10  # Minimum risk
            risk_factors.append("Baseline assessment")
        
        # Determine risk level
        if risk_score >= 70:
            risk_level = 'HIGH'
        elif risk_score >= 40:
            risk_level = 'MODERATE'
        else:
            risk_level = 'LOW'
        
        assessment_result = {
            'risk_level': risk_level,
            'risk_probability': min(risk_score / 100, 0.95),
            'confidence_score': 0.8,
            'risk_factors': risk_factors
        }
        
        # Generate AI report
        if gemini_model:
            try:
                prompt = f"""
Generate a comprehensive medical report for a patient with {condition} risk assessment.

Patient Data: {json.dumps(patient_data, indent=2)}
Risk Level: {assessment_result['risk_level']}
Risk Probability: {assessment_result['risk_probability']:.1%}

Please provide a professional medical report with:
1. Executive Summary
2. Clinical Analysis
3. Recommendations
4. Follow-up Care

Format professionally for medical documentation.
"""
                response = gemini_model.generate_content(prompt)
                ai_report = response.text
            except Exception as e:
                ai_report = generate_fallback_report(patient_data, assessment_result, condition)
        else:
            ai_report = generate_fallback_report(patient_data, assessment_result, condition)
        
        return jsonify({
            'success': True,
            'report': ai_report,
            'assessment': assessment_result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_fallback_report(patient_data, assessment_result, condition):
    """Generate fallback report without external AI"""
    return f"""
# COMPREHENSIVE {condition.upper()} RISK ASSESSMENT REPORT

## EXECUTIVE SUMMARY
Patient Risk Level: {assessment_result['risk_level']}
Risk Probability: {assessment_result['risk_probability']:.1%}
Confidence Score: {assessment_result['confidence_score']:.1%}

## CLINICAL ANALYSIS
Based on the provided patient data, a comprehensive assessment has been performed.

## RECOMMENDATIONS
1. Schedule follow-up consultation
2. Implement lifestyle modifications
3. Regular monitoring as indicated

## FOLLOW-UP CARE
- Regular health checkups
- Monitor key indicators
- Patient education and counseling

---
Report Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Risk Assessment System: Wellora AI Healthcare Platform
"""

@app.route('/api/dashboard-data')
def dashboard_data():
    """API endpoint for dashboard data"""
    try:
        # Calculate statistics
        total_assessments = assessment_storage['total_count']
        risk_distribution = assessment_storage['risk_counts']
        condition_breakdown = assessment_storage['condition_counts']
        recent_assessments = assessment_storage['assessments'][-10:]  # Last 10
        
        return jsonify({
            'total_assessments': total_assessments,
            'risk_distribution': risk_distribution,
            'condition_breakdown': condition_breakdown,
            'recent_assessments': recent_assessments
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint for Render deployment"""
    return jsonify({
        'status': 'healthy',
        'service': 'Wellora AI Healthcare Platform',
        'version': '1.0.0',
        'timestamp': __import__('datetime').datetime.now().isoformat()
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') != 'production'
    
    print("=" * 60)
    print("WELLORA AI HEALTHCARE PLATFORM")
    print("=" * 60)
    print(f"Starting web server on port {port}")
    print(f"Debug mode: {debug}")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=debug)

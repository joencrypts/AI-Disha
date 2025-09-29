"""
AI-Powered Proactive Patient Risk Advisor - Flask Application
Integrates Meditex template with AI healthcare functionality
"""

import sys
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
import pandas as pd
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def generate_ai_report(patient_data, assessment_result, condition):
    """Generate comprehensive AI report using Gemini or fallback"""
    if gemini_model is None:
        return generate_fallback_report(patient_data, assessment_result, condition)
    
    try:
        # Prepare data for Gemini
        ml_predictions = assessment_result.get('ml_predictions', {})
        risk_level = assessment_result.get('risk_level', 'UNKNOWN')
        risk_probability = assessment_result.get('risk_probability', 0.5)
        confidence_score = assessment_result.get('confidence_score', 0.5)
        
        # Create comprehensive prompt
        prompt = f"""
You are a medical AI assistant analyzing patient data for {condition} risk assessment. 
Generate a comprehensive medical report based on the following data:

PATIENT DATA:
{json.dumps(patient_data, indent=2)}

ASSESSMENT RESULTS:
- Risk Level: {risk_level}
- Risk Probability: {risk_probability:.3f} ({risk_probability*100:.1f}%)
- Confidence Score: {confidence_score:.3f} ({confidence_score*100:.1f}%)

MACHINE LEARNING PREDICTIONS:
{json.dumps(ml_predictions, indent=2)}

Please generate a professional medical report that includes:

1. EXECUTIVE SUMMARY
   - Overall risk assessment
   - Key findings
   - Urgency level

2. CLINICAL ANALYSIS
   - Detailed analysis of patient data
   - Risk factors identified
   - Clinical significance of findings

3. AI MODEL INSIGHTS
   - Analysis of ML predictions
   - Model agreement/disagreement
   - Confidence assessment

4. RECOMMENDATIONS
   - Immediate actions required
   - Follow-up care needed
   - Lifestyle modifications
   - Monitoring requirements

5. CLINICAL NOTES
   - Important considerations
   - Red flags to watch
   - Patient counseling points

Format the report professionally with clear sections and medical terminology. 
Be specific about risk levels and provide actionable recommendations.
"""

        # Generate report using Gemini
        response = gemini_model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        # Fallback to local report generation
        return generate_fallback_report(patient_data, assessment_result, condition)

def generate_fallback_report(patient_data, assessment_result, condition):
    """Generate comprehensive report without external AI"""
    risk_level = assessment_result.get('risk_level', 'UNKNOWN')
    risk_probability = assessment_result.get('risk_probability', 0.5)
    confidence_score = assessment_result.get('confidence_score', 0.5)
    ml_predictions = assessment_result.get('ml_predictions', {})
    
    # Analyze patient data
    age = patient_data.get('age', 'N/A')
    gender = 'Male' if patient_data.get('gender') == 1 else 'Female'
    
    # Condition-specific analysis
    if condition == 'diabetes':
        bmi = patient_data.get('bmi', 'N/A')
        hba1c = patient_data.get('HbA1c_level', 'N/A')
        glucose = patient_data.get('blood_glucose_level', 'N/A')
        hypertension = 'Yes' if patient_data.get('hypertension') == 1 else 'No'
        heart_disease = 'Yes' if patient_data.get('heart_disease') == 1 else 'No'
        smoking = patient_data.get('smoking_history', 'N/A')
        
        report = f"""
# COMPREHENSIVE DIABETES RISK ASSESSMENT REPORT

## EXECUTIVE SUMMARY
Patient: {age}-year-old {gender}
Risk Level: {risk_level}
Risk Probability: {risk_probability:.1%}
Confidence Score: {confidence_score:.1%}

## CLINICAL ANALYSIS

### Patient Demographics
- Age: {age} years
- Gender: {gender}
- BMI: {bmi} kg/m²

### Key Risk Factors
- HbA1c Level: {hba1c}% {'(Very High Risk)' if isinstance(hba1c, (int, float)) and hba1c > 7 else '(Normal)' if isinstance(hba1c, (int, float)) and hba1c < 5.7 else ''}
- Blood Glucose: {glucose} mg/dL {'(Very High Risk)' if isinstance(glucose, (int, float)) and glucose > 200 else '(Normal)' if isinstance(glucose, (int, float)) and glucose < 100 else ''}
- Hypertension: {hypertension}
- Heart Disease History: {heart_disease}
- Smoking Status: {smoking}

### Clinical Significance
Based on the patient data, {'multiple high-risk factors are present' if risk_level == 'HIGH' else 'moderate risk factors are identified' if risk_level == 'MODERATE' else 'low risk factors are present'}.

## AI MODEL INSIGHTS

### Machine Learning Predictions
"""
        
        for model, pred in ml_predictions.items():
            if isinstance(pred, dict) and 'probability' in pred:
                prob = pred['probability']
                report += f"- {model.replace('_', ' ').title()}: {prob:.1%}\n"
        
        report += f"""
### Model Analysis
The AI models show {'strong agreement' if confidence_score > 0.7 else 'moderate agreement' if confidence_score > 0.5 else 'some disagreement'} in their predictions.

## RECOMMENDATIONS

### Immediate Actions
"""
        
        if risk_level == 'HIGH':
            report += """
- URGENT: Schedule immediate consultation with endocrinologist
- Begin intensive glucose monitoring
- Consider immediate medication intervention
- Implement strict dietary modifications
"""
        elif risk_level == 'MODERATE':
            report += """
- Schedule appointment with primary care physician within 1 week
- Begin regular glucose monitoring
- Implement lifestyle modifications
- Consider medication evaluation
"""
        else:
            report += """
- Continue regular monitoring
- Maintain healthy lifestyle
- Annual diabetes screening
- Preventive care measures
"""
        
        report += f"""
### Follow-up Care
- Regular HbA1c testing every 3-6 months
- Annual comprehensive diabetes evaluation
- Regular eye and foot examinations
- Blood pressure monitoring

### Lifestyle Modifications
- Dietary counseling for diabetes management
- Regular physical activity program
- Weight management if BMI > 25
- Smoking cessation if applicable

## CLINICAL NOTES

### Important Considerations
- Patient education on diabetes management
- Medication adherence counseling
- Regular monitoring of complications
- Family history assessment

### Red Flags to Watch
- Rapidly increasing glucose levels
- Development of complications
- Medication side effects
- Lifestyle compliance issues

### Patient Counseling Points
- Importance of regular monitoring
- Dietary modifications
- Exercise recommendations
- Medication compliance
- Early warning signs of complications

---
Report Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Risk Assessment System: Wellora AI Healthcare Platform
"""
        
        return report

def store_assessment_result(condition, result):
    """Store real assessment result for dashboard analytics"""
    import datetime
    
    # Create assessment record
    assessment = {
        'id': f'P{assessment_storage["total_count"] + 1:03d}',
        'condition': condition,
        'risk_level': result['risk_level'],
        'risk_probability': result['risk_probability'],
        'confidence_score': result['confidence_score'],
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ml_predictions': result.get('ml_predictions', {})
    }
    
    # Store assessment
    assessment_storage['assessments'].append(assessment)
    assessment_storage['total_count'] += 1
    
    # Update counters
    risk_level = result['risk_level']
    if risk_level in assessment_storage['risk_counts']:
        assessment_storage['risk_counts'][risk_level] += 1
    
    if condition in assessment_storage['condition_counts']:
        assessment_storage['condition_counts'][condition] += 1
    
    # Keep only last 100 assessments to prevent memory issues
    if len(assessment_storage['assessments']) > 100:
        assessment_storage['assessments'] = assessment_storage['assessments'][-100:]

def apply_feature_engineering(patient_data, condition):
    """Apply feature engineering to match training data"""
    if condition == 'kidney':
        # For kidney disease, use the exact fields from the original dataset
        # Remove gender field as it's not in the original dataset
        df_data = {}
        for key, value in patient_data.items():
            if key.lower() != 'gender':  # Skip gender field
                df_data[key.lower()] = value
        
        df = pd.DataFrame([df_data])
        
        # Add missing features that were in the original dataset
        if 'id' not in df.columns:
            df['id'] = 1  # Dummy ID
        if 'sg' not in df.columns:
            df['sg'] = 1.02  # Default specific gravity
        if 'al' not in df.columns:
            df['al'] = 0  # Default albumin
        if 'su' not in df.columns:
            df['su'] = 0  # Default sugar
        if 'rbc' not in df.columns:
            df['rbc'] = 0  # Default RBC
        if 'pc' not in df.columns:
            df['pc'] = 0  # Default pus cell
        if 'pcc' not in df.columns:
            df['pcc'] = 0  # Default pus cell clumps
        if 'ba' not in df.columns:
            df['ba'] = 0  # Default bacteria
        if 'cad' not in df.columns:
            df['cad'] = 0  # Default coronary artery disease
        if 'appet' not in df.columns:
            df['appet'] = 0  # Default appetite
        if 'pe' not in df.columns:
            df['pe'] = 0  # Default pedal edema
        if 'ane' not in df.columns:
            df['ane'] = 0  # Default anemia
        
        # Add engineered features that were created during training
        df['Cardiovascular_Risk_Score'] = 0  # Default risk score
        df['Kidney_Risk_Score'] = 0  # Default risk score
        
    elif condition == 'diabetes':
        # For diabetes, use the exact fields from the original dataset
        df_data = {}
        for key, value in patient_data.items():
            df_data[key.lower()] = value
        
        df = pd.DataFrame([df_data])
        
        # Handle gender encoding (convert 0/1 to Female/Male, then to 0/1 as in training)
        if 'gender' in df.columns:
            # Convert 0/1 to Female/Male, then back to 0/1 as in training
            df['gender'] = df['gender'].map({0: 'Female', 1: 'Male'})
            df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
            df['gender'] = df['gender'].fillna(0)  # Default to Female (0)
        
        # Handle smoking history encoding (same as in training)
        if 'smoking_history' in df.columns:
            smoking_mapping = {
                'never': 0,
                'No Info': 1,
                'former': 2,
                'current': 3,
                'not current': 2,
                'ever': 2
            }
            df['smoking_history'] = df['smoking_history'].map(smoking_mapping)
            df['smoking_history'] = df['smoking_history'].fillna(1)  # Fill unknown with 'No Info'
        
        # Add engineered features that were created during training
        df['Cardiovascular_Risk_Score'] = 0  # Default risk score
        df['Diabetes_Flag'] = 0  # Default flag
        df['Diabetes_Risk_Score'] = 0  # Default risk score
        
        df['diabetes'] = 0  # This is the target, set to 0 for prediction
        
    elif condition == 'heart':
        # For heart disease, use the exact fields from the original dataset
        df_data = {}
        for key, value in patient_data.items():
            df_data[key.lower()] = value
        
        df = pd.DataFrame([df_data])
        df['Heart Disease'] = 0  # This is the target, set to 0 for prediction
        
    elif condition == 'liver':
        # For liver disease, use the exact fields from the original dataset
        df_data = {}
        for key, value in patient_data.items():
            df_data[key.lower()] = value
        
        df = pd.DataFrame([df_data])
        df['Result'] = 0  # This is the target, set to 0 for prediction
    
    return df.iloc[0].to_dict()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Production configuration
if os.getenv('FLASK_ENV') == 'production':
    app.config['DEBUG'] = False
else:
    app.config['DEBUG'] = True

# Initialize system
try:
    from main import AIHealthcareSystem
    system = AIHealthcareSystem()
    system.setup_knowledge_base()
    print("Wellora system initialized successfully!")
except Exception as e:
    print(f"Error initializing system: {e}")
    system = None

# Initialize assessment storage
assessment_storage = {
    'assessments': [],
    'total_count': 0,
    'risk_counts': {'HIGH': 0, 'MODERATE': 0, 'LOW': 0},
    'condition_counts': {'kidney': 0, 'diabetes': 0, 'heart': 0, 'liver': 0}
}

# Configure Gemini AI
try:
    # Using environment variable for API key
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

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    """Contact page"""
    if request.method == 'POST':
        # Handle contact form submission
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        subject = request.form.get('subject')
        message = request.form.get('message')
        
        # Here you would typically save to database or send email
        # For now, just return success
        return jsonify({'success': True, 'message': 'Thank you for your message!'})
    
    return render_template('contact.html')

@app.route('/risk-assessment')
def risk_assessment():
    """Risk assessment page"""
    return render_template('risk_assessment.html')

@app.route('/assess_risk', methods=['POST'])
def assess_risk():
    """Assess patient risk"""
    try:
        condition = request.form.get('condition')
        patient_data = {}
        
        # Get form data - collect all fields based on condition
        if request.form.get('age'):
            patient_data['Age'] = int(request.form.get('age'))
        if request.form.get('gender'):
            patient_data['Gender'] = int(request.form.get('gender'))
        
        # Collect condition-specific fields
        if condition == 'kidney':
            # Kidney disease specific fields
            if request.form.get('bp'):
                patient_data['bp'] = float(request.form.get('bp'))
            if request.form.get('bgr'):
                patient_data['bgr'] = float(request.form.get('bgr'))
            if request.form.get('bu'):
                patient_data['bu'] = float(request.form.get('bu'))
            if request.form.get('sc'):
                patient_data['sc'] = float(request.form.get('sc'))
            if request.form.get('sod'):
                patient_data['sod'] = float(request.form.get('sod'))
            if request.form.get('pot'):
                patient_data['pot'] = float(request.form.get('pot'))
            if request.form.get('hemo'):
                patient_data['hemo'] = float(request.form.get('hemo'))
            if request.form.get('pcv'):
                patient_data['pcv'] = float(request.form.get('pcv'))
            if request.form.get('wc'):
                patient_data['wc'] = float(request.form.get('wc'))
            if request.form.get('rc'):
                patient_data['rc'] = float(request.form.get('rc'))
            if request.form.get('htn'):
                # Convert yes/no to 1/0
                patient_data['htn'] = 1 if request.form.get('htn') == 'yes' else 0
            if request.form.get('dm'):
                # Convert yes/no to 1/0
                patient_data['dm'] = 1 if request.form.get('dm') == 'yes' else 0
                
        elif condition == 'diabetes':
            # Diabetes specific fields
            if request.form.get('bmi'):
                patient_data['bmi'] = float(request.form.get('bmi'))
            if request.form.get('hypertension'):
                patient_data['hypertension'] = int(request.form.get('hypertension'))
            if request.form.get('heart_disease'):
                patient_data['heart_disease'] = int(request.form.get('heart_disease'))
            if request.form.get('smoking_history'):
                patient_data['smoking_history'] = request.form.get('smoking_history')
            if request.form.get('HbA1c_level'):
                patient_data['HbA1c_level'] = float(request.form.get('HbA1c_level'))
            if request.form.get('blood_glucose_level'):
                patient_data['blood_glucose_level'] = int(request.form.get('blood_glucose_level'))
                
        elif condition == 'heart':
            # Heart disease specific fields
            if request.form.get('chest_pain_type'):
                patient_data['chest_pain_type'] = int(request.form.get('chest_pain_type'))
            if request.form.get('bp'):
                patient_data['bp'] = float(request.form.get('bp'))
            if request.form.get('cholesterol'):
                patient_data['cholesterol'] = int(request.form.get('cholesterol'))
            if request.form.get('fbs_over_120'):
                patient_data['fbs_over_120'] = int(request.form.get('fbs_over_120'))
            if request.form.get('ekg_results'):
                patient_data['ekg_results'] = int(request.form.get('ekg_results'))
            if request.form.get('max_hr'):
                patient_data['max_hr'] = int(request.form.get('max_hr'))
            if request.form.get('exercise_angina'):
                patient_data['exercise_angina'] = int(request.form.get('exercise_angina'))
            if request.form.get('st_depression'):
                patient_data['st_depression'] = float(request.form.get('st_depression'))
            if request.form.get('slope_st'):
                patient_data['slope_st'] = int(request.form.get('slope_st'))
                
        elif condition == 'liver':
            # Liver disease specific fields
            if request.form.get('total_bilirubin'):
                patient_data['total_bilirubin'] = float(request.form.get('total_bilirubin'))
            if request.form.get('direct_bilirubin'):
                patient_data['direct_bilirubin'] = float(request.form.get('direct_bilirubin'))
            if request.form.get('alkaline_phosphatase'):
                patient_data['alkaline_phosphatase'] = int(request.form.get('alkaline_phosphatase'))
            if request.form.get('alt'):
                patient_data['alt'] = int(request.form.get('alt'))
            if request.form.get('ast'):
                patient_data['ast'] = int(request.form.get('ast'))
            if request.form.get('total_proteins'):
                patient_data['total_proteins'] = float(request.form.get('total_proteins'))
            if request.form.get('albumin'):
                patient_data['albumin'] = float(request.form.get('albumin'))
            if request.form.get('ag_ratio'):
                patient_data['ag_ratio'] = float(request.form.get('ag_ratio'))
        
        # Handle JSON input
        if request.form.get('patient_data'):
            try:
                json_data = json.loads(request.form.get('patient_data'))
                patient_data.update(json_data)
            except json.JSONDecodeError:
                return jsonify({'error': 'Invalid JSON format'}), 400
        
        # Feature engineering is now handled by the hybrid inference system
        
        if not system:
            return jsonify({'error': 'System not initialized'}), 500
        
        # Use hybrid inference system for comprehensive risk assessment
        try:
            # Setup hybrid system if not already done
            if system.hybrid_system is None:
                system.setup_hybrid_system()
            
            # Make hybrid prediction
            hybrid_result = system.predict_patient_risk(patient_data, condition)
            
            result = {
                'risk_level': hybrid_result['final_risk_level'],
                'risk_probability': hybrid_result['final_risk_probability'],
                'confidence_score': hybrid_result['confidence_score'],
                'matched_rules': hybrid_result['kbs_results']['matched_rules'],
                'matched_rules_count': len(hybrid_result['kbs_results']['matched_rules']),
                'recommended_actions': hybrid_result['recommended_actions'],
                'explanation': hybrid_result['explanation'],
                'ml_predictions': hybrid_result['ml_predictions'],
                'kbs_results': hybrid_result['kbs_results']
            }
            
            # Store real assessment data for dashboard
            store_assessment_result(condition, result)
        except Exception as e:
            # Fallback to knowledge base only if hybrid system fails
            print(f"Hybrid system failed, using knowledge base only: {e}")
            matched_rules = system.knowledge_base.evaluate_rules(patient_data, condition)
            general_rules = system.knowledge_base.evaluate_rules(patient_data, 'general')
            all_rules = matched_rules + general_rules
            
            risk_level = system.knowledge_base.get_risk_level(all_rules)
            explanation = system.knowledge_base.generate_explanation(all_rules, risk_level)
            recommended_actions = system.knowledge_base.get_recommended_actions(all_rules)
            
            # Calculate risk probability based on matched rules
            risk_probability = 0.2  # Default low risk
            if risk_level == 'HIGH':
                risk_probability = 0.8
            elif risk_level == 'MODERATE':
                risk_probability = 0.5
            
            result = {
                'risk_level': risk_level,
                'risk_probability': risk_probability,
                'confidence_score': 0.75,  # Default confidence
                'matched_rules': all_rules,
                'matched_rules_count': len(all_rules),
                'recommended_actions': recommended_actions,
                'explanation': explanation
            }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/assess_bulk_risk', methods=['POST'])
def assess_bulk_risk():
    """Assess multiple patients at once"""
    try:
        condition = request.form.get('condition')
        bulk_data_text = request.form.get('bulk_data')
        
        try:
            bulk_data = json.loads(bulk_data_text)
            if not isinstance(bulk_data, list):
                return jsonify({'error': 'Bulk data must be a JSON array'}), 400
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid JSON format'}), 400
        
        if not system:
            return jsonify({'error': 'System not initialized'}), 500
        
        results = []
        for patient_data in bulk_data:
            # Use knowledge base to assess risk
            matched_rules = system.knowledge_base.evaluate_rules(patient_data, condition)
            general_rules = system.knowledge_base.evaluate_rules(patient_data, 'general')
            all_rules = matched_rules + general_rules
            
            risk_level = system.knowledge_base.get_risk_level(all_rules)
            explanation = system.knowledge_base.generate_explanation(all_rules, risk_level)
            recommended_actions = system.knowledge_base.get_recommended_actions(all_rules)
            
            # Calculate risk probability based on matched rules
            risk_probability = 0.2  # Default low risk
            if risk_level == 'HIGH':
                risk_probability = 0.8
            elif risk_level == 'MODERATE':
                risk_probability = 0.5
            
            result = {
                'risk_level': risk_level,
                'risk_probability': risk_probability,
                'confidence_score': 0.75,  # Default confidence
                'matched_rules': all_rules,
                'matched_rules_count': len(all_rules),
                'recommended_actions': recommended_actions,
                'explanation': explanation
            }
            results.append(result)
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/quick_assessment', methods=['POST'])
def quick_assessment():
    """Quick assessment from homepage form"""
    try:
        name = request.form.get('name')
        email = request.form.get('email')
        condition = request.form.get('condition')
        age = int(request.form.get('age'))
        
        # Create basic patient data
        patient_data = {
            'Age': age,
            'Gender': 1,  # Default to male
            'BMI': 25.0,  # Default BMI
            'SystolicBP': 120,  # Default BP
            'DiastolicBP': 80
        }
        
        if not system:
            flash('System not available. Please try again later.', 'error')
            return redirect(url_for('index'))
        
        # Use knowledge base to assess risk
        matched_rules = system.knowledge_base.evaluate_rules(patient_data, condition)
        general_rules = system.knowledge_base.evaluate_rules(patient_data, 'general')
        all_rules = matched_rules + general_rules
        
        risk_level = system.knowledge_base.get_risk_level(all_rules)
        
        flash(f'Quick Assessment Complete: {risk_level} risk detected. Please visit our detailed assessment page for comprehensive analysis.', 'info')
        return redirect(url_for('risk_assessment'))
        
    except Exception as e:
        flash(f'Error in quick assessment: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/departments')
def departments():
    """All departments page"""
    return render_template('departments.html')

@app.route('/departments/kidney')
def kidney_department():
    """Kidney disease department"""
    return render_template('department.html', condition='kidney', title='Kidney Disease')

@app.route('/departments/diabetes')
def diabetes_department():
    """Diabetes department"""
    return render_template('department.html', condition='diabetes', title='Diabetes')

@app.route('/departments/heart')
def heart_department():
    """Heart disease department"""
    return render_template('department.html', condition='heart', title='Heart Disease')

@app.route('/departments/liver')
def liver_department():
    """Liver disease department"""
    return render_template('department.html', condition='liver', title='Liver Disease')

@app.route('/bulk-assessment')
def bulk_assessment():
    """Bulk assessment page"""
    return redirect(url_for('risk_assessment'))

@app.route('/clinician-dashboard')
def clinician_dashboard():
    """Clinician dashboard"""
    if system:
        try:
            system.run_dashboard(port=8050)
            return redirect('http://localhost:8050')
        except Exception as e:
            flash(f'Error starting dashboard: {str(e)}', 'error')
            return redirect(url_for('index'))
    else:
        flash('System not available. Please try again later.', 'error')
        return redirect(url_for('index'))


@app.route('/doctor-dashboard')
def doctor_dashboard():
    """Doctor dashboard page"""
    return render_template('doctor_dashboard.html')

@app.route('/faq')
def faq():
    """FAQ page"""
    return render_template('faq.html')

@app.route('/search')
def search():
    """Search functionality"""
    query = request.args.get('q', '')
    # Implement search functionality here
    return render_template('search_results.html', query=query)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'system': 'AI Healthcare System'})

@app.route('/api/dashboard-data')
def dashboard_data():
    """Get dashboard data with real statistics from actual assessments"""
    try:
        # Calculate real statistics from stored assessments
        total_assessments = assessment_storage['total_count']
        high_risk = assessment_storage['risk_counts']['HIGH']
        moderate_risk = assessment_storage['risk_counts']['MODERATE']
        low_risk = assessment_storage['risk_counts']['LOW']
        
        # Calculate average risk probability
        if assessment_storage['assessments']:
            avg_probability = sum(a['risk_probability'] for a in assessment_storage['assessments']) / len(assessment_storage['assessments'])
            avg_probability = avg_probability * 100  # Convert to percentage
        else:
            avg_probability = 0
        
        # Get recent assessments (last 10)
        recent_assessments = assessment_storage['assessments'][-10:] if assessment_storage['assessments'] else []
        
        dashboard_data = {
            'kpis': {
                'total': total_assessments,
                'high': high_risk,
                'avgProb': round(avg_probability, 1),
                'completed': total_assessments  # All assessments are completed
            },
            'riskDistribution': {
                'labels': ['High', 'Moderate', 'Low'],
                'values': [high_risk, moderate_risk, low_risk]
            },
            'conditionBreakdown': {
                'labels': ['Kidney', 'Diabetes', 'Heart', 'Liver'],
                'values': [
                    assessment_storage['condition_counts']['kidney'],
                    assessment_storage['condition_counts']['diabetes'],
                    assessment_storage['condition_counts']['heart'],
                    assessment_storage['condition_counts']['liver']
                ]
            },
            'recent': [
                {
                    'id': a['id'],
                    'condition': a['condition'],
                    'risk': a['risk_level'],
                    'prob': round(a['risk_probability'], 2),
                    'conf': round(a['confidence_score'], 2),
                    'timestamp': a['timestamp']
                }
                for a in recent_assessments
            ]
        }
        
        return jsonify(dashboard_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_report', methods=['POST'])
def generate_report():
    """Generate AI report using Gemini"""
    try:
        # Get form data
        condition = request.form.get('condition')
        patient_data = {}
        
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
        
        if not system:
            return jsonify({'error': 'System not initialized'}), 500
        
        # Perform assessment first
        try:
            if system.hybrid_system is None:
                system.setup_hybrid_system()
            
            hybrid_result = system.predict_patient_risk(patient_data, condition)
            
            assessment_result = {
                'risk_level': hybrid_result['final_risk_level'],
                'risk_probability': hybrid_result['final_risk_probability'],
                'confidence_score': hybrid_result['confidence_score'],
                'ml_predictions': hybrid_result['ml_predictions']
            }
            
            # Generate AI report
            ai_report = generate_ai_report(patient_data, assessment_result, condition)
            
            return jsonify({
                'success': True,
                'report': ai_report,
                'assessment': assessment_result
            })
            
        except Exception as e:
            return jsonify({'error': f'Assessment failed: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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

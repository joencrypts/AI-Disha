"""
Simple web server for AI-Powered Proactive Patient Risk Advisor
Provides a basic web interface to interact with the system
"""

import sys
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify
import json

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

app = Flask(__name__)

# Initialize system
try:
    from main import AIHealthcareSystem
    system = AIHealthcareSystem()
    system.setup_knowledge_base()
    print("AI Healthcare System initialized successfully!")
except Exception as e:
    print(f"Error initializing system: {e}")
    system = None

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI-Powered Proactive Patient Risk Advisor</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #1a1a1a;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
            background: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        
        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        
        .header p {
            color: #666;
            font-size: 1.1rem;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 30px 60px rgba(0,0,0,0.15);
        }
        
        .card h2 {
            color: #1a1a1a;
            margin-bottom: 20px;
            font-size: 1.5rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .form-group {
            margin-bottom: 25px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #1a1a1a;
            font-size: 0.95rem;
        }
        
        .input-group {
            position: relative;
        }
        
        .input-group i {
            position: absolute;
            left: 15px;
            top: 50%;
            transform: translateY(-50%);
            color: #667eea;
            z-index: 1;
        }
        
        input, select, textarea {
            width: 100%;
            padding: 15px 15px 15px 45px;
            border: 2px solid #e1e5e9;
            border-radius: 12px;
            font-size: 16px;
            transition: all 0.3s ease;
            background: #f8f9fa;
            color: #1a1a1a;
        }
        
        input:focus, select:focus, textarea:focus {
            outline: none;
            border-color: #667eea;
            background: white;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .button-group {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 8px;
            text-decoration: none;
            position: relative;
            overflow: hidden;
        }
        
        .btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
        }
        
        .btn:hover::before {
            left: 100%;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        }
        
        .btn-secondary {
            background: linear-gradient(135deg, #f093fb, #f5576c);
            color: white;
            box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);
        }
        
        .btn-secondary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(240, 147, 251, 0.4);
        }
        
        .btn-success {
            background: linear-gradient(135deg, #4facfe, #00f2fe);
            color: white;
            box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
        }
        
        .btn-success:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(79, 172, 254, 0.4);
        }
        
        .btn-outline {
            background: transparent;
            color: #667eea;
            border: 2px solid #667eea;
        }
        
        .btn-outline:hover {
            background: #667eea;
            color: white;
            transform: translateY(-2px);
        }
        
        .data-input-section {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            margin-bottom: 30px;
        }
        
        .input-tabs {
            display: flex;
            margin-bottom: 20px;
            background: #f8f9fa;
            border-radius: 12px;
            padding: 5px;
        }
        
        .tab-button {
            flex: 1;
            padding: 12px 20px;
            border: none;
            background: transparent;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 600;
            color: #666;
        }
        
        .tab-button.active {
            background: white;
            color: #667eea;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .manual-input-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        
        .result {
            margin-top: 30px;
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
        }
        
        .result.risk-high {
            background: linear-gradient(135deg, #ff6b6b, #ee5a52);
            color: white;
        }
        
        .result.risk-moderate {
            background: linear-gradient(135deg, #feca57, #ff9ff3);
            color: #1a1a1a;
        }
        
        .result.risk-low {
            background: linear-gradient(135deg, #48dbfb, #0abde3);
            color: white;
        }
        
        .risk-level {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .risk-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .metric-card {
            background: rgba(255, 255, 255, 0.2);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 5px;
        }
        
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.8;
        }
        
        .risk-factors, .recommendations {
            margin: 20px 0;
        }
        
        .risk-factor, .recommendation {
            background: rgba(255, 255, 255, 0.2);
            padding: 15px;
            margin: 10px 0;
            border-radius: 12px;
            border-left: 4px solid rgba(255, 255, 255, 0.5);
            transition: all 0.3s ease;
        }
        
        .risk-factor:hover, .recommendation:hover {
            transform: translateX(5px);
            background: rgba(255, 255, 255, 0.3);
        }
        
        .sample-patients {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        
        .patient-card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .patient-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        }
        
        .patient-info {
            font-weight: 700;
            color: #1a1a1a;
            margin-bottom: 15px;
            font-size: 1.1rem;
        }
        
        .patient-details {
            color: #666;
            margin-bottom: 15px;
        }
        
        .risk-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .risk-badge.high {
            background: #ff6b6b;
            color: white;
        }
        
        .risk-badge.moderate {
            background: #feca57;
            color: #1a1a1a;
        }
        
        .risk-badge.low {
            background: #48dbfb;
            color: white;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .fade-in {
            animation: fadeIn 0.5s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
            
            .manual-input-grid {
                grid-template-columns: 1fr;
            }
            
            .button-group {
                flex-direction: column;
            }
            
            .header h1 {
                font-size: 2rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-heartbeat"></i> AI-Powered Proactive Patient Risk Advisor</h1>
            <p>Advanced healthcare risk assessment powered by machine learning and clinical knowledge</p>
        </div>
        
        <div class="data-input-section">
            <h2><i class="fas fa-user-plus"></i> Patient Data Entry</h2>
            
            <div class="input-tabs">
                <button class="tab-button active" onclick="switchTab('manual')">
                    <i class="fas fa-keyboard"></i> Manual Input
                </button>
                <button class="tab-button" onclick="switchTab('json')">
                    <i class="fas fa-code"></i> JSON Input
                </button>
                <button class="tab-button" onclick="switchTab('bulk')">
                    <i class="fas fa-upload"></i> Bulk Upload
                </button>
            </div>
            
            <div id="manual-tab" class="tab-content active">
                <div class="form-group">
                    <label for="condition">Condition Type:</label>
                    <div class="input-group">
                        <i class="fas fa-stethoscope"></i>
                        <select id="condition" name="condition">
                            <option value="kidney">Kidney Disease</option>
                            <option value="diabetes">Diabetes</option>
                            <option value="heart">Heart Disease</option>
                            <option value="liver">Liver Disease</option>
                        </select>
                    </div>
                </div>
                
                <div class="manual-input-grid">
                    <div class="form-group">
                        <label for="age">Age:</label>
                        <div class="input-group">
                            <i class="fas fa-calendar"></i>
                            <input type="number" id="age" name="age" placeholder="Enter age" min="0" max="120">
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label for="gender">Gender:</label>
                        <div class="input-group">
                            <i class="fas fa-venus-mars"></i>
                            <select id="gender" name="gender">
                                <option value="1">Male</option>
                                <option value="0">Female</option>
                            </select>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label for="bmi">BMI:</label>
                        <div class="input-group">
                            <i class="fas fa-weight"></i>
                            <input type="number" id="bmi" name="bmi" placeholder="Enter BMI" step="0.1" min="10" max="100">
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label for="systolic_bp">Systolic BP:</label>
                        <div class="input-group">
                            <i class="fas fa-heartbeat"></i>
                            <input type="number" id="systolic_bp" name="systolic_bp" placeholder="Enter systolic BP" min="50" max="300">
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label for="diastolic_bp">Diastolic BP:</label>
                        <div class="input-group">
                            <i class="fas fa-heartbeat"></i>
                            <input type="number" id="diastolic_bp" name="diastolic_bp" placeholder="Enter diastolic BP" min="30" max="200">
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label for="creatinine">Serum Creatinine:</label>
                        <div class="input-group">
                            <i class="fas fa-flask"></i>
                            <input type="number" id="creatinine" name="creatinine" placeholder="Enter creatinine level" step="0.1" min="0" max="20">
                        </div>
                    </div>
                </div>
            </div>
            
            <div id="json-tab" class="tab-content">
                <div class="form-group">
                    <label for="condition_json">Condition Type:</label>
                    <div class="input-group">
                        <i class="fas fa-stethoscope"></i>
                        <select id="condition_json" name="condition_json">
                            <option value="kidney">Kidney Disease</option>
                            <option value="diabetes">Diabetes</option>
                            <option value="heart">Heart Disease</option>
                            <option value="liver">Liver Disease</option>
                        </select>
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="patient_data">Patient Data (JSON format):</label>
                    <textarea id="patient_data" name="patient_data" rows="12" placeholder='Enter patient data in JSON format, e.g.:
{
    "Age": 65,
    "BMI": 28.5,
    "SystolicBP": 145,
    "DiastolicBP": 95,
    "SerumCreatinine": 2.1,
    "eGFR": 25,
    "Gender": 1
}'></textarea>
                </div>
            </div>
            
            <div id="bulk-tab" class="tab-content">
                <div class="form-group">
                    <label for="bulk_condition">Condition Type:</label>
                    <div class="input-group">
                        <i class="fas fa-stethoscope"></i>
                        <select id="bulk_condition" name="bulk_condition">
                            <option value="kidney">Kidney Disease</option>
                            <option value="diabetes">Diabetes</option>
                            <option value="heart">Heart Disease</option>
                            <option value="liver">Liver Disease</option>
                        </select>
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="bulk_data">Bulk Patient Data (JSON Array):</label>
                    <textarea id="bulk_data" name="bulk_data" rows="15" placeholder='Enter multiple patients in JSON array format, e.g.:
[
    {
        "Age": 65,
        "BMI": 28.5,
        "SystolicBP": 145,
        "DiastolicBP": 95,
        "SerumCreatinine": 2.1,
        "eGFR": 25,
        "Gender": 1
    },
    {
        "Age": 45,
        "BMI": 32.1,
        "SystolicBP": 130,
        "DiastolicBP": 85,
        "SerumCreatinine": 1.2,
        "eGFR": 60,
        "Gender": 0
    }
]'></textarea>
                </div>
            </div>
            
            <div class="button-group">
                <button class="btn btn-primary" onclick="assessRisk()">
                    <i class="fas fa-search"></i> Assess Patient Risk
                </button>
                <button class="btn btn-secondary" onclick="loadSamplePatient()">
                    <i class="fas fa-user-md"></i> Load Sample Patient
                </button>
                <button class="btn btn-success" onclick="assessBulkRisk()">
                    <i class="fas fa-users"></i> Assess Bulk Patients
                </button>
                <button class="btn btn-outline" onclick="clearResults()">
                    <i class="fas fa-trash"></i> Clear Results
                </button>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing patient data...</p>
            </div>
        </div>
        
        <div id="result" class="result" style="display: none;">
            <div class="risk-level" id="risk-level"></div>
            <div class="risk-metrics" id="risk-metrics"></div>
            <div id="result-content"></div>
        </div>
        
        <div class="sample-patients">
            <h2><i class="fas fa-users"></i> Sample Patient Scenarios</h2>
            <div class="patient-card" onclick="loadSamplePatient('kidney')">
                <div class="patient-info">High-Risk Kidney Patient (John Smith, 72M)</div>
                <div class="patient-details">eGFR: 25, Systolic BP: 145, Creatinine: 2.8</div>
                <span class="risk-badge high">HIGH RISK</span>
            </div>
            
            <div class="patient-card" onclick="loadSamplePatient('diabetes')">
                <div class="patient-info">Moderate-Risk Diabetes Patient (Sarah Johnson, 45F)</div>
                <div class="patient-details">BMI: 32.1, HbA1c: 7.8, Glucose: 180</div>
                <span class="risk-badge moderate">MODERATE RISK</span>
            </div>
            
            <div class="patient-card" onclick="loadSamplePatient('heart')">
                <div class="patient-info">High-Risk Heart Patient (Robert Davis, 58M)</div>
                <div class="patient-details">Cholesterol: 280, BP: 160, Chest Pain: Yes</div>
                <span class="risk-badge high">HIGH RISK</span>
            </div>
            
            <div class="patient-card" onclick="loadSamplePatient('liver')">
                <div class="patient-info">Moderate-Risk Liver Patient (Maria Garcia, 55F)</div>
                <div class="patient-details">Bilirubin: 2.5, AST: 120, Albumin: 2.8</div>
                <span class="risk-badge moderate">MODERATE RISK</span>
            </div>
        </div>
    </div>

    <script>
        // Tab switching functionality
        function switchTab(tabName) {
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Remove active class from all tab buttons
            document.querySelectorAll('.tab-button').forEach(button => {
                button.classList.remove('active');
            });
            
            // Show selected tab content
            document.getElementById(tabName + '-tab').classList.add('active');
            
            // Add active class to clicked button
            event.target.classList.add('active');
        }
        
        // Sample patient data
        const samplePatients = {
            kidney: {
                condition: 'kidney',
                data: {
                    "Age": 72,
                    "Gender": 1,
                    "BMI": 28.5,
                    "SystolicBP": 145,
                    "DiastolicBP": 95,
                    "SerumCreatinine": 2.8,
                    "eGFR": 25,
                    "ProteinInUrine": 450,
                    "ACR": 350,
                    "CholesterolTotal": 220,
                    "CholesterolLDL": 140,
                    "CholesterolHDL": 35,
                    "Hypertension_Flag": 1,
                    "Diabetes_Flag": 0,
                    "Obesity_Flag": 0
                }
            },
            diabetes: {
                condition: 'diabetes',
                data: {
                    "age": 45,
                    "gender": "Female",
                    "bmi": 32.1,
                    "hypertension": 1,
                    "heart_disease": 0,
                    "smoking_history": "former",
                    "HbA1c_level": 7.8,
                    "blood_glucose_level": 180,
                    "family_history": 1
                }
            },
            heart: {
                condition: 'heart',
                data: {
                    "Age": 58,
                    "Sex": 1,
                    "Chest pain type": 3,
                    "BP": 160,
                    "Cholesterol": 280,
                    "FBS over 120": 1,
                    "EKG results": 2,
                    "Max HR": 140,
                    "Exercise angina": 1,
                    "ST depression": 1.8,
                    "Slope of ST": 2,
                    "Number of vessels fluro": 2,
                    "Thallium": 6
                }
            },
            liver: {
                condition: 'liver',
                data: {
                    "Age of the patient": 55,
                    "Gender of the patient": "Female",
                    "Total Bilirubin": 2.5,
                    "Direct Bilirubin": 1.2,
                    "Alkphos Alkaline Phosphotase": 450,
                    "Sgpt Alamine Aminotransferase": 85,
                    "Sgot Aspartate Aminotransferase": 120,
                    "Total Protiens": 6.2,
                    "ALB Albumin": 2.8,
                    "A/G Ratio Albumin and Globulin Ratio": 0.8
                }
            }
        };
        
        function loadSamplePatient(type = 'kidney') {
            const patient = samplePatients[type];
            if (!patient) return;
            
            // Switch to JSON tab
            switchTab('json');
            
            // Set condition
            document.getElementById('condition_json').value = patient.condition;
            
            // Set patient data
            document.getElementById('patient_data').value = JSON.stringify(patient.data, null, 2);
            
            // Also populate manual inputs
            populateManualInputs(patient.data);
        }
        
        function populateManualInputs(data) {
            // Map common fields
            if (data.Age) document.getElementById('age').value = data.Age;
            if (data.age) document.getElementById('age').value = data.age;
            if (data['Age of the patient']) document.getElementById('age').value = data['Age of the patient'];
            
            if (data.Gender !== undefined) document.getElementById('gender').value = data.Gender;
            if (data.gender === 'Female') document.getElementById('gender').value = 0;
            if (data.gender === 'Male') document.getElementById('gender').value = 1;
            if (data['Gender of the patient'] === 'Female') document.getElementById('gender').value = 0;
            if (data['Gender of the patient'] === 'Male') document.getElementById('gender').value = 1;
            
            if (data.BMI) document.getElementById('bmi').value = data.BMI;
            if (data.bmi) document.getElementById('bmi').value = data.bmi;
            
            if (data.SystolicBP) document.getElementById('systolic_bp').value = data.SystolicBP;
            if (data.BP) document.getElementById('systolic_bp').value = data.BP;
            
            if (data.DiastolicBP) document.getElementById('diastolic_bp').value = data.DiastolicBP;
            
            if (data.SerumCreatinine) document.getElementById('creatinine').value = data.SerumCreatinine;
        }
        
        function getPatientDataFromForm() {
            const condition = document.getElementById('condition').value || document.getElementById('condition_json').value;
            const patientData = {};
            
            // Get manual input data
            const age = document.getElementById('age').value;
            const gender = document.getElementById('gender').value;
            const bmi = document.getElementById('bmi').value;
            const systolicBp = document.getElementById('systolic_bp').value;
            const diastolicBp = document.getElementById('diastolic_bp').value;
            const creatinine = document.getElementById('creatinine').value;
            
            if (age) patientData.Age = parseInt(age);
            if (gender) patientData.Gender = parseInt(gender);
            if (bmi) patientData.BMI = parseFloat(bmi);
            if (systolicBp) patientData.SystolicBP = parseInt(systolicBp);
            if (diastolicBp) patientData.DiastolicBP = parseInt(diastolicBp);
            if (creatinine) patientData.SerumCreatinine = parseFloat(creatinine);
            
            return { condition, patientData };
        }
        
        function showLoading() {
            document.getElementById('loading').style.display = 'block';
        }
        
        function hideLoading() {
            document.getElementById('loading').style.display = 'none';
        }
        
        function assessRisk() {
            showLoading();
            
            let condition, patientData;
            
            // Check which tab is active
            const activeTab = document.querySelector('.tab-content.active');
            
            if (activeTab.id === 'manual-tab') {
                const formData = getPatientDataFromForm();
                condition = formData.condition;
                patientData = formData.patientData;
            } else if (activeTab.id === 'json-tab') {
                condition = document.getElementById('condition_json').value;
                const patientDataText = document.getElementById('patient_data').value;
                
                try {
                    patientData = JSON.parse(patientDataText);
                } catch (error) {
                    hideLoading();
                    alert('Invalid JSON format. Please check your patient data.');
                    return;
                }
            } else {
                hideLoading();
                alert('Please select a valid input method.');
                return;
            }
            
            fetch('/assess_risk', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    condition: condition,
                    patient_data: patientData
                })
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                displayResult(data);
            })
            .catch(error => {
                hideLoading();
                console.error('Error:', error);
                alert('Error assessing risk: ' + error.message);
            });
        }
        
        function assessBulkRisk() {
            const condition = document.getElementById('bulk_condition').value;
            const bulkDataText = document.getElementById('bulk_data').value;
            
            try {
                const bulkData = JSON.parse(bulkDataText);
                if (!Array.isArray(bulkData)) {
                    alert('Bulk data must be a JSON array.');
                    return;
                }
                
                showLoading();
                
                // Process each patient
                Promise.all(bulkData.map(patientData => 
                    fetch('/assess_risk', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            condition: condition,
                            patient_data: patientData
                        })
                    }).then(response => response.json())
                ))
                .then(results => {
                    hideLoading();
                    displayBulkResults(results);
                })
                .catch(error => {
                    hideLoading();
                    console.error('Error:', error);
                    alert('Error assessing bulk patients: ' + error.message);
                });
                
            } catch (error) {
                alert('Invalid JSON format. Please check your bulk patient data.');
            }
        }
        
        function displayResult(data) {
            const resultDiv = document.getElementById('result');
            const contentDiv = document.getElementById('result-content');
            const riskLevelDiv = document.getElementById('risk-level');
            const riskMetricsDiv = document.getElementById('risk-metrics');
            
            let riskClass = 'risk-low';
            if (data.risk_level === 'HIGH') riskClass = 'risk-high';
            else if (data.risk_level === 'MODERATE') riskClass = 'risk-moderate';
            
            riskLevelDiv.innerHTML = `
                <i class="fas fa-exclamation-triangle"></i> ${data.risk_level} RISK
            `;
            
            riskMetricsDiv.innerHTML = `
                <div class="metric-card">
                    <div class="metric-value">${(data.risk_probability * 100).toFixed(1)}%</div>
                    <div class="metric-label">Risk Probability</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${(data.confidence_score * 100).toFixed(1)}%</div>
                    <div class="metric-label">Confidence Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${data.matched_rules_count}</div>
                    <div class="metric-label">Matched Rules</div>
                </div>
            `;
            
            contentDiv.innerHTML = `
                <div class="risk-factors">
                    <h4><i class="fas fa-exclamation-circle"></i> Matched Clinical Rules:</h4>
                    ${data.matched_rules.map(rule => `
                        <div class="risk-factor">
                            <strong>${rule.name}</strong><br>
                            ${rule.explanation}
                        </div>
                    `).join('')}
                </div>
                
                <div class="recommendations">
                    <h4><i class="fas fa-clipboard-list"></i> Recommended Actions:</h4>
                    ${data.recommended_actions.map(action => `
                        <div class="recommendation">${action}</div>
                    `).join('')}
                </div>
                
                <div style="margin-top: 20px;">
                    <h4><i class="fas fa-info-circle"></i> Detailed Explanation:</h4>
                    <p>${data.explanation}</p>
                </div>
            `;
            
            resultDiv.className = `result ${riskClass} fade-in`;
            resultDiv.style.display = 'block';
        }
        
        function displayBulkResults(results) {
            const resultDiv = document.getElementById('result');
            const contentDiv = document.getElementById('result-content');
            const riskLevelDiv = document.getElementById('risk-level');
            const riskMetricsDiv = document.getElementById('risk-metrics');
            
            // Calculate summary statistics
            const highRisk = results.filter(r => r.risk_level === 'HIGH').length;
            const moderateRisk = results.filter(r => r.risk_level === 'MODERATE').length;
            const lowRisk = results.filter(r => r.risk_level === 'LOW').length;
            const avgProbability = results.reduce((sum, r) => sum + r.risk_probability, 0) / results.length;
            const avgConfidence = results.reduce((sum, r) => sum + r.confidence_score, 0) / results.length;
            
            riskLevelDiv.innerHTML = `
                <i class="fas fa-users"></i> BULK ASSESSMENT RESULTS
            `;
            
            riskMetricsDiv.innerHTML = `
                <div class="metric-card">
                    <div class="metric-value">${results.length}</div>
                    <div class="metric-label">Total Patients</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${highRisk}</div>
                    <div class="metric-label">High Risk</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${moderateRisk}</div>
                    <div class="metric-label">Moderate Risk</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${lowRisk}</div>
                    <div class="metric-label">Low Risk</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${(avgProbability * 100).toFixed(1)}%</div>
                    <div class="metric-label">Avg Probability</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${(avgConfidence * 100).toFixed(1)}%</div>
                    <div class="metric-label">Avg Confidence</div>
                </div>
            `;
            
            contentDiv.innerHTML = `
                <div class="risk-factors">
                    <h4><i class="fas fa-chart-bar"></i> Risk Distribution Summary:</h4>
                    <div class="risk-factor">
                        <strong>High Risk Patients (${highRisk}):</strong> Require immediate medical attention and specialist consultation.
                    </div>
                    <div class="risk-factor">
                        <strong>Moderate Risk Patients (${moderateRisk}):</strong> Need regular monitoring and lifestyle modifications.
                    </div>
                    <div class="risk-factor">
                        <strong>Low Risk Patients (${lowRisk}):</strong> Continue with regular preventive care.
                    </div>
                </div>
                
                <div class="recommendations">
                    <h4><i class="fas fa-clipboard-list"></i> Overall Recommendations:</h4>
                    <div class="recommendation">Prioritize high-risk patients for immediate consultation</div>
                    <div class="recommendation">Schedule follow-up appointments for moderate-risk patients</div>
                    <div class="recommendation">Continue routine monitoring for low-risk patients</div>
                    <div class="recommendation">Consider population-level health interventions based on risk distribution</div>
                </div>
            `;
            
            resultDiv.className = `result risk-moderate fade-in`;
            resultDiv.style.display = 'block';
        }
        
        function clearResults() {
            document.getElementById('result').style.display = 'none';
            document.getElementById('patient_data').value = '';
            document.getElementById('bulk_data').value = '';
            
            // Clear manual inputs
            document.getElementById('age').value = '';
            document.getElementById('bmi').value = '';
            document.getElementById('systolic_bp').value = '';
            document.getElementById('diastolic_bp').value = '';
            document.getElementById('creatinine').value = '';
        }
        
        // Initialize the page
        document.addEventListener('DOMContentLoaded', function() {
            // Add smooth scrolling
            document.querySelectorAll('a[href^="#"]').forEach(anchor => {
                anchor.addEventListener('click', function (e) {
                    e.preventDefault();
                    document.querySelector(this.getAttribute('href')).scrollIntoView({
                        behavior: 'smooth'
                    });
                });
            });
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/assess_risk', methods=['POST'])
def assess_risk():
    """Assess patient risk"""
    try:
        data = request.get_json()
        condition = data.get('condition')
        patient_data = data.get('patient_data')
        
        if not system:
            return jsonify({'error': 'System not initialized'}), 500
        
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
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/assess_bulk_risk', methods=['POST'])
def assess_bulk_risk():
    """Assess multiple patients at once"""
    try:
        data = request.get_json()
        condition = data.get('condition')
        patients = data.get('patients', [])
        
        if not system:
            return jsonify({'error': 'System not initialized'}), 500
        
        results = []
        for patient_data in patients:
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

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'system': 'AI Healthcare System'})

if __name__ == '__main__':
    print("="*60)
    print("AI-POWERED PROACTIVE PATIENT RISK ADVISOR")
    print("WEB SERVER")
    print("="*60)
    print("Starting web server...")
    print("Dashboard will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

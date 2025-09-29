"""
Sample Patient Scenarios for AI-Powered Proactive Patient Risk Advisor
Demonstrates the system with realistic patient cases
"""

import json
from pathlib import Path

def create_sample_scenarios():
    """Create comprehensive sample patient scenarios"""
    
    scenarios = {
        "high_risk_kidney_patient": {
            "patient_id": "P001",
            "name": "John Smith",
            "age": 72,
            "gender": "Male",
            "condition_type": "kidney",
            "clinical_data": {
                "Age": 72,
                "Gender": 1,  # Male
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
                "Obesity_Flag": 0,
                "FamilyHistoryKidneyDisease": 1,
                "PreviousAcuteKidneyInjury": 0
            },
            "symptoms": [
                "Fatigue and weakness",
                "Swelling in feet and ankles",
                "Changes in urination frequency",
                "Loss of appetite",
                "Nausea and vomiting"
            ],
            "expected_risk_level": "HIGH",
            "expected_explanations": [
                "Severe CKD by eGFR - eGFR below 30 indicates severe kidney dysfunction",
                "Proteinuria Risk - Elevated protein in urine indicates kidney damage",
                "Hypertension in CKD - High blood pressure accelerates kidney damage"
            ],
            "expected_actions": [
                "Immediate nephrology consultation",
                "ACE inhibitor therapy",
                "Blood pressure management",
                "Regular monitoring"
            ]
        },
        
        "moderate_risk_diabetes_patient": {
            "patient_id": "P002",
            "name": "Sarah Johnson",
            "age": 45,
            "gender": "Female",
            "condition_type": "diabetes",
            "clinical_data": {
                "age": 45,
                "gender": "Female",
                "bmi": 32.1,
                "hypertension": 1,
                "heart_disease": 0,
                "smoking_history": "former",
                "HbA1c_level": 7.8,
                "blood_glucose_level": 180,
                "family_history": 1
            },
            "symptoms": [
                "Increased thirst and urination",
                "Unexplained weight loss",
                "Fatigue and blurred vision",
                "Slow healing of cuts and bruises"
            ],
            "expected_risk_level": "MODERATE",
            "expected_explanations": [
                "Poor Glycemic Control - HbA1c above 8.0 indicates poor diabetes management",
                "Diabetes with Obesity - BMI above 30 complicates diabetes management"
            ],
            "expected_actions": [
                "Intensify diabetes management",
                "Weight management program",
                "Regular physical activity",
                "Nutritional counseling"
            ]
        },
        
        "high_risk_heart_patient": {
            "patient_id": "P003",
            "name": "Robert Davis",
            "age": 58,
            "gender": "Male",
            "condition_type": "heart",
            "clinical_data": {
                "Age": 58,
                "Sex": 1,  # Male
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
            },
            "symptoms": [
                "Chest pain and pressure",
                "Shortness of breath during activity",
                "Irregular heartbeat",
                "Dizziness and fatigue"
            ],
            "expected_risk_level": "HIGH",
            "expected_explanations": [
                "High Cholesterol Risk - Cholesterol above 240 requires immediate intervention",
                "Hypertension Risk - Blood pressure above 140/90 is a major risk factor",
                "Multiple Risk Factors - Age, cholesterol, and diabetes increase risk significantly"
            ],
            "expected_actions": [
                "High-intensity statin therapy",
                "Cardiology consultation",
                "Blood pressure management",
                "Lifestyle modifications"
            ]
        },
        
        "moderate_risk_liver_patient": {
            "patient_id": "P004",
            "name": "Maria Garcia",
            "age": 55,
            "gender": "Female",
            "condition_type": "liver",
            "clinical_data": {
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
            },
            "symptoms": [
                "Abdominal pain and discomfort",
                "Yellowing of skin and eyes",
                "Loss of appetite and weight",
                "Fatigue and weakness"
            ],
            "expected_risk_level": "MODERATE",
            "expected_explanations": [
                "Moderate Liver Dysfunction - Bilirubin and albumin levels indicate liver stress",
                "Elevated Liver Enzymes - AST and ALT levels suggest liver damage"
            ],
            "expected_actions": [
                "Regular liver function tests",
                "Hepatology consultation",
                "Avoid alcohol and hepatotoxic substances",
                "Monitor for complications"
            ]
        },
        
        "low_risk_healthy_patient": {
            "patient_id": "P005",
            "name": "David Wilson",
            "age": 35,
            "gender": "Male",
            "condition_type": "general",
            "clinical_data": {
                "Age": 35,
                "Gender": 1,  # Male
                "BMI": 24.2,
                "SystolicBP": 118,
                "DiastolicBP": 78,
                "CholesterolTotal": 180,
                "CholesterolLDL": 110,
                "CholesterolHDL": 55,
                "Smoking": 0,
                "Diabetes": 0,
                "Hypertension": 0
            },
            "symptoms": [
                "No significant symptoms",
                "Regular exercise routine",
                "Healthy diet",
                "Regular check-ups"
            ],
            "expected_risk_level": "LOW",
            "expected_explanations": [
                "No significant risk factors identified",
                "Healthy lifestyle indicators present"
            ],
            "expected_actions": [
                "Continue regular monitoring",
                "Maintain healthy lifestyle",
                "Annual preventive care"
            ]
        }
    }
    
    return scenarios

def create_voice_interaction_examples():
    """Create example voice interactions for each scenario"""
    
    voice_examples = {
        "high_risk_kidney_patient": {
            "initial_script": """
            Hello! I'm your AI healthcare assistant. I'm here to help assess your kidney health risk.
            
            Let me ask you a few questions to better understand your situation:
            
            1. Have you been experiencing any changes in your urination patterns recently?
            2. Do you have any swelling in your feet, ankles, or hands?
            3. Have you noticed any changes in your energy levels or fatigue?
            4. Are you currently taking any medications for blood pressure or diabetes?
            5. Do you have a family history of kidney disease?
            
            Please share any symptoms or concerns you have, and I'll provide personalized recommendations based on your responses.
            """,
            "response_script": """
            Thank you for sharing your symptoms and concerns. Based on my analysis, here's what I found:
            
            RISK ASSESSMENT:
            Your current risk level is HIGH. The probability of developing kidney disease is 85%.
            
            ⚠️ HIGH RISK ALERT ⚠️
            Your symptoms and risk factors indicate a high risk. I strongly recommend:
            - Seeking immediate medical attention
            - Consulting with a specialist
            - Following up with your primary care physician
            
            RECOMMENDED ACTIONS:
            1. Immediate nephrology consultation
            2. ACE inhibitor therapy
            3. Blood pressure management
            4. Regular monitoring
            
            IMPORTANT DISCLAIMER:
            This assessment is for informational purposes only and should not replace professional medical advice. 
            Please consult with a qualified healthcare provider for proper diagnosis and treatment.
            
            Would you like me to explain any of these recommendations in more detail?
            """
        },
        
        "moderate_risk_diabetes_patient": {
            "initial_script": """
            Hello! I'm your AI healthcare assistant. I'm here to help assess your diabetes risk.
            
            Let me ask you a few questions to better understand your situation:
            
            1. Have you been experiencing increased thirst or frequent urination?
            2. Have you noticed any unexplained weight changes recently?
            3. Do you have any family history of diabetes?
            4. How would you describe your current diet and exercise routine?
            5. Have you had any recent blood sugar tests?
            
            Please share any symptoms or concerns you have, and I'll provide personalized recommendations based on your responses.
            """,
            "response_script": """
            Thank you for sharing your symptoms and concerns. Based on my analysis, here's what I found:
            
            RISK ASSESSMENT:
            Your current risk level is MODERATE. The probability of developing diabetes complications is 65%.
            
            ⚠️ MODERATE RISK
            Your symptoms suggest moderate risk. I recommend:
            - Scheduling an appointment with your doctor
            - Monitoring your symptoms closely
            - Making lifestyle modifications
            
            RECOMMENDED ACTIONS:
            1. Intensify diabetes management
            2. Weight management program
            3. Regular physical activity
            4. Nutritional counseling
            
            IMPORTANT DISCLAIMER:
            This assessment is for informational purposes only and should not replace professional medical advice. 
            Please consult with a qualified healthcare provider for proper diagnosis and treatment.
            
            Would you like me to explain any of these recommendations in more detail?
            """
        }
    }
    
    return voice_examples

def create_dashboard_examples():
    """Create example dashboard displays for each scenario"""
    
    dashboard_examples = {
        "high_risk_kidney_patient": {
            "risk_card": {
                "patient_id": "P001",
                "name": "John Smith",
                "age": 72,
                "condition": "Chronic Kidney Disease",
                "risk_level": "HIGH",
                "risk_probability": 0.85,
                "confidence_score": 0.78,
                "last_updated": "2024-01-15 14:30",
                "key_risk_factors": [
                    "eGFR < 30 (Severe kidney dysfunction)",
                    "Proteinuria > 300 mg/g",
                    "Hypertension (BP > 140/90)",
                    "Age > 65 years"
                ],
                "recommended_actions": [
                    "Immediate nephrology consultation",
                    "ACE inhibitor therapy",
                    "Blood pressure management",
                    "Regular monitoring"
                ]
            },
            "lab_trends": {
                "eGFR": [45, 38, 32, 28, 25],
                "Creatinine": [1.8, 2.1, 2.4, 2.6, 2.8],
                "Proteinuria": [200, 280, 350, 400, 450],
                "BUN": [25, 30, 35, 40, 45]
            }
        },
        
        "moderate_risk_diabetes_patient": {
            "risk_card": {
                "patient_id": "P002",
                "name": "Sarah Johnson",
                "age": 45,
                "condition": "Diabetes Mellitus",
                "risk_level": "MODERATE",
                "risk_probability": 0.65,
                "confidence_score": 0.72,
                "last_updated": "2024-01-15 14:30",
                "key_risk_factors": [
                    "HbA1c > 7.5 (Poor glycemic control)",
                    "BMI > 30 (Obesity)",
                    "Family history of diabetes",
                    "Previous gestational diabetes"
                ],
                "recommended_actions": [
                    "Intensify diabetes management",
                    "Weight management program",
                    "Regular physical activity",
                    "Nutritional counseling"
                ]
            },
            "lab_trends": {
                "HbA1c": [7.2, 7.5, 7.8, 7.6, 7.8],
                "Glucose": [150, 165, 180, 175, 180],
                "BMI": [31.5, 31.8, 32.1, 32.0, 32.1]
            }
        }
    }
    
    return dashboard_examples

def save_sample_scenarios():
    """Save all sample scenarios to files"""
    
    # Create scenarios
    scenarios = create_sample_scenarios()
    voice_examples = create_voice_interaction_examples()
    dashboard_examples = create_dashboard_examples()
    
    # Save to JSON files
    scenarios_path = Path("sample_patient_scenarios.json")
    with open(scenarios_path, 'w') as f:
        json.dump(scenarios, f, indent=2)
    
    voice_path = Path("sample_voice_interactions.json")
    with open(voice_path, 'w') as f:
        json.dump(voice_examples, f, indent=2)
    
    dashboard_path = Path("sample_dashboard_examples.json")
    with open(dashboard_path, 'w') as f:
        json.dump(dashboard_examples, f, indent=2)
    
    print(f"Sample scenarios saved to: {scenarios_path}")
    print(f"Voice examples saved to: {voice_path}")
    print(f"Dashboard examples saved to: {dashboard_path}")
    
    return scenarios, voice_examples, dashboard_examples

if __name__ == "__main__":
    scenarios, voice_examples, dashboard_examples = save_sample_scenarios()
    
    print("\n=== SAMPLE PATIENT SCENARIOS ===")
    for scenario_id, scenario in scenarios.items():
        print(f"\n{scenario_id.upper()}:")
        print(f"  Patient: {scenario['name']} (ID: {scenario['patient_id']})")
        print(f"  Condition: {scenario['condition_type'].title()}")
        print(f"  Expected Risk: {scenario['expected_risk_level']}")
        print(f"  Key Symptoms: {', '.join(scenario['symptoms'][:3])}...")
    
    print("\n=== VOICE INTERACTION EXAMPLES ===")
    for example_id, example in voice_examples.items():
        print(f"\n{example_id.upper()}:")
        print(f"  Initial Script Length: {len(example['initial_script'])} characters")
        print(f"  Response Script Length: {len(example['response_script'])} characters")
    
    print("\n=== DASHBOARD EXAMPLES ===")
    for example_id, example in dashboard_examples.items():
        print(f"\n{example_id.upper()}:")
        print(f"  Risk Level: {example['risk_card']['risk_level']}")
        print(f"  Risk Probability: {example['risk_card']['risk_probability']}")
        print(f"  Lab Trends Available: {len(example['lab_trends'])} parameters")

"""
Demonstration script for AI-Powered Proactive Patient Risk Advisor
Shows complete system functionality with sample data
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def run_demo():
    """Run complete system demonstration"""
    
    print("="*80)
    print("AI-POWERED PROACTIVE PATIENT RISK ADVISOR - DEMONSTRATION")
    print("="*80)
    
    try:
        # Import main system
        from main import AIHealthcareSystem
        
        # Initialize system
        print("\n1. Initializing AI Healthcare System...")
        system = AIHealthcareSystem()
        
        # Setup knowledge base
        print("\n2. Setting up Knowledge Base...")
        system.setup_knowledge_base()
        
        # Process one dataset as demonstration
        print("\n3. Processing Kidney Disease Dataset...")
        kidney_results = system.process_dataset('kidney')
        
        if kidney_results:
            print("✓ Kidney dataset processed successfully")
            print(f"  - Data shape: {kidney_results['data'].shape}")
            print(f"  - Models trained: {len(kidney_results['models'])}")
            print(f"  - Features created: {len(kidney_results['feature_names'])}")
        else:
            print("✗ Failed to process kidney dataset")
            return
        
        # Setup hybrid system
        print("\n4. Setting up Hybrid Inference System...")
        system.setup_hybrid_system()
        
        # Create sample patient scenarios
        print("\n5. Creating Sample Patient Scenarios...")
        from sample_patient_scenarios import create_sample_scenarios
        scenarios = create_sample_scenarios()
        
        # Demonstrate predictions
        print("\n6. Demonstrating Patient Risk Predictions...")
        print("-" * 60)
        
        for scenario_id, scenario in scenarios.items():
            print(f"\n--- {scenario_id.upper().replace('_', ' ')} ---")
            print(f"Patient: {scenario['name']} (ID: {scenario['patient_id']})")
            print(f"Condition: {scenario['condition_type'].title()}")
            print(f"Age: {scenario['age']}, Gender: {scenario['gender']}")
            
            # Make prediction
            try:
                prediction = system.predict_patient_risk(
                    scenario['clinical_data'], 
                    scenario['condition_type']
                )
                
                print(f"Risk Level: {prediction['final_risk_level']}")
                print(f"Risk Probability: {prediction['final_risk_probability']:.3f}")
                print(f"Confidence Score: {prediction['confidence_score']:.3f}")
                print(f"ML Models Used: {len(prediction['ml_predictions'])}")
                print(f"Clinical Rules Matched: {len(prediction['kbs_results']['matched_rules'])}")
                
                # Show key risk factors
                if prediction['kbs_results']['matched_rules']:
                    print("Key Risk Factors:")
                    for i, rule in enumerate(prediction['kbs_results']['matched_rules'][:3], 1):
                        print(f"  {i}. {rule['name']} - {rule['explanation']}")
                
                # Show recommended actions
                if prediction['recommended_actions']:
                    print("Recommended Actions:")
                    for i, action in enumerate(prediction['recommended_actions'][:3], 1):
                        print(f"  {i}. {action}")
                
            except Exception as e:
                print(f"Error making prediction: {e}")
            
            print("-" * 60)
        
        # Demonstrate voice assistant
        print("\n7. Demonstrating Voice Assistant...")
        try:
            # Use high-risk kidney patient for voice demo
            kidney_patient = scenarios['high_risk_kidney_patient']
            prediction = system.predict_patient_risk(
                kidney_patient['clinical_data'], 
                'kidney'
            )
            
            voice_interaction = system.create_voice_interaction(
                'kidney',
                "Patient reports fatigue, swelling, and changes in urination",
                prediction
            )
            
            print("Voice Interaction Created:")
            print(f"  - Initial Script: {len(voice_interaction['initial_script'])} characters")
            print(f"  - Response Script: {len(voice_interaction['response_script'])} characters")
            print(f"  - Audio Files: {voice_interaction['initial_audio_path']}")
            
        except Exception as e:
            print(f"Voice assistant demo error: {e}")
        
        # Show evaluation results
        print("\n8. Model Evaluation Results...")
        if 'evaluation' in kidney_results:
            eval_results = kidney_results['evaluation']
            print("Model Performance Summary:")
            for model_name, results in eval_results.items():
                print(f"  {model_name}:")
                print(f"    - AUC-ROC: {results['auc_roc']:.3f}")
                print(f"    - F1-Score: {results['f1_score']:.3f}")
                print(f"    - Precision: {results['precision']:.3f}")
                print(f"    - Recall: {results['recall']:.3f}")
        
        # Show file structure
        print("\n9. Generated Files and Directories...")
        results_dir = Path("results")
        if results_dir.exists():
            print("Results directory structure:")
            for item in results_dir.rglob("*"):
                if item.is_file():
                    print(f"  {item.relative_to(results_dir)}")
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print("\nNext Steps:")
        print("1. Run the clinician dashboard: system.run_dashboard()")
        print("2. Process other datasets: system.process_dataset('diabetes')")
        print("3. View generated visualizations in results/ directory")
        print("4. Check sample scenarios in sample_patient_scenarios.json")
        
        return system, kidney_results
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def show_system_capabilities():
    """Show system capabilities and features"""
    
    print("\n" + "="*80)
    print("SYSTEM CAPABILITIES OVERVIEW")
    print("="*80)
    
    capabilities = {
        "Data Processing": [
            "Missing value handling with auto-strategy",
            "Outlier detection using Isolation Forest",
            "Feature normalization and encoding",
            "Data validation and quality checks"
        ],
        "Feature Engineering": [
            "BMI categories based on WHO standards",
            "Blood pressure categories (AHA guidelines)",
            "eGFR calculation using MDRD formula",
            "Liver function ratios (AST/ALT, Bilirubin/Albumin)",
            "Cholesterol ratios (LDL/HDL, Total/HDL)",
            "Comorbidity flags and risk scores"
        ],
        "Machine Learning": [
            "Logistic Regression with balanced weights",
            "XGBoost with hyperparameter optimization",
            "LightGBM with categorical support",
            "Stratified K-fold cross-validation",
            "Model calibration and probability adjustment"
        ],
        "Explainability": [
            "SHAP explanations (Tree, Linear, Kernel)",
            "LIME local interpretations",
            "Feature importance ranking",
            "Waterfall plots for individual predictions",
            "Model comparison and analysis"
        ],
        "Knowledge Base": [
            "Evidence-based clinical rules",
            "Risk stratification (Low/Moderate/High)",
            "Actionable recommendations",
            "Medical literature references",
            "Rule evaluation and matching"
        ],
        "Hybrid Inference": [
            "ML prediction combination",
            "KBS rule integration",
            "Confidence scoring",
            "Risk level determination",
            "Comprehensive explanations"
        ],
        "Voice Assistant": [
            "ElevenLabs voice synthesis",
            "Natural language interaction",
            "Symptom collection and response",
            "Emergency alert generation",
            "Audio file creation and management"
        ],
        "Clinician Dashboard": [
            "Interactive web interface",
            "Patient risk cards",
            "Population risk analysis",
            "Lab trend visualization",
            "Model performance metrics",
            "Voice integration"
        ],
        "Evaluation": [
            "AUC-ROC and PR-AUC analysis",
            "Sensitivity, specificity, F1-score",
            "Calibration assessment",
            "Decision curve analysis",
            "Confusion matrix visualization",
            "Model comparison metrics"
        ]
    }
    
    for category, features in capabilities.items():
        print(f"\n{category}:")
        for feature in features:
            print(f"  ✓ {feature}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    # Show system capabilities
    show_system_capabilities()
    
    # Run demonstration
    system, results = run_demo()
    
    if system:
        print("\nSystem is ready for use!")
        print("To run the dashboard: system.run_dashboard(port=8050)")
        print("To process more datasets: system.process_all_datasets()")
    else:
        print("\nDemo failed. Please check the error messages above.")

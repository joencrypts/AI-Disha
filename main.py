"""
Main application file for AI-Powered Proactive Patient Risk Advisor
Integrates all components and provides end-to-end functionality
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import all modules
from config import *
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from ml_models import MLModelTrainer
from explainability import ModelExplainer
from knowledge_base import KnowledgeBaseSystem
from hybrid_inference import HybridInferenceSystem
from voice_assistant import VoiceAssistant
from clinician_dashboard import ClinicianDashboard
from evaluation import ModelEvaluator

class AIHealthcareSystem:
    """Main AI-Powered Proactive Patient Risk Advisor system"""
    
    def __init__(self):
        self.config = {
            'DATA_DIR': DATA_DIR,
            'MODELS_DIR': MODELS_DIR,
            'RESULTS_DIR': RESULTS_DIR,
            'STATIC_DIR': STATIC_DIR,
            'DATASETS': DATASETS,
            'MODEL_CONFIG': MODEL_CONFIG,
            'FEATURE_CONFIG': FEATURE_CONFIG,
            'RISK_LEVELS': RISK_LEVELS,
            'ELEVENLABS_CONFIG': ELEVENLABS_CONFIG,
            'CLINICAL_THRESHOLDS': CLINICAL_THRESHOLDS
        }
        
        # Initialize components
        self.preprocessor = DataPreprocessor(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.ml_trainer = MLModelTrainer(self.config)
        self.explainer = ModelExplainer(self.config)
        self.knowledge_base = KnowledgeBaseSystem(self.config)
        self.voice_assistant = VoiceAssistant(self.config)
        self.evaluator = ModelEvaluator(self.config)
        
        # Initialize hybrid system
        self.hybrid_system = None
        
        print("AI-Powered Proactive Patient Risk Advisor initialized successfully!")
    
    def setup_knowledge_base(self):
        """Setup knowledge base with clinical rules"""
        print("\n=== Setting up Knowledge Base ===")
        
        # Create clinical rules
        self.knowledge_base.create_clinical_rules()
        
        # Save rules to JSON
        rules_path = self.config['RESULTS_DIR'] / 'clinical_rules.json'
        self.knowledge_base.save_rules_to_json(rules_path)
        
        print("Knowledge base setup completed!")
    
    def process_dataset(self, condition_type):
        """Process a single dataset through the complete pipeline"""
        print(f"\n=== Processing {condition_type} dataset ===")
        
        # Load and preprocess data
        df = self.preprocessor.load_dataset(condition_type)
        if df is None:
            return None
        
        # Preprocess data
        df_processed = self.preprocessor.preprocess_dataset(
            condition_type, 
            target_col=DATASETS[condition_type]['target']
        )
        
        # Feature engineering
        df_engineered = self.feature_engineer.engineer_features(
            df_processed, 
            condition_type
        )
        
        # Train ML models
        target_col = DATASETS[condition_type]['target']
        models_results, X_test, y_test, feature_names = self.ml_trainer.train_all_models(
            df_engineered, 
            target_col, 
            condition_type
        )
        
        # Save models
        self.ml_trainer.save_models(condition_type, models_results)
        
        # Evaluate models
        evaluation_results = self.evaluator.evaluate_all_models(
            models_results, X_test, y_test, condition_type
        )
        
        # Save evaluation results
        self.evaluator.save_evaluation_results(
            evaluation_results, condition_type, self.config['RESULTS_DIR']
        )
        
        # Create visualizations
        self._create_evaluation_visualizations(evaluation_results, condition_type)
        
        return {
            'data': df_engineered,
            'models': models_results,
            'evaluation': evaluation_results,
            'feature_names': feature_names
        }
    
    def _create_evaluation_visualizations(self, evaluation_results, condition_type):
        """Create evaluation visualizations"""
        print(f"Creating evaluation visualizations for {condition_type}")
        
        viz_dir = self.config['RESULTS_DIR'] / condition_type / 'visualizations'
        viz_dir.mkdir(exist_ok=True, parents=True)
        
        # ROC curves
        roc_fig = self.evaluator.create_roc_curves_plot(
            evaluation_results, condition_type, 
            save_path=viz_dir / 'roc_curves.html'
        )
        
        # Precision-Recall curves
        pr_fig = self.evaluator.create_precision_recall_curves_plot(
            evaluation_results, condition_type,
            save_path=viz_dir / 'precision_recall_curves.html'
        )
        
        # Calibration plots
        cal_fig = self.evaluator.create_calibration_plot(
            evaluation_results, condition_type,
            save_path=viz_dir / 'calibration_plot.html'
        )
        
        # Confusion matrices
        cm_fig = self.evaluator.create_confusion_matrices_plot(
            evaluation_results, condition_type,
            save_path=viz_dir / 'confusion_matrices.html'
        )
        
        # Metrics comparison
        metrics_fig = self.evaluator.create_metrics_comparison_plot(
            evaluation_results, condition_type,
            save_path=viz_dir / 'metrics_comparison.html'
        )
        
        # Decision curve analysis
        dca_fig = self.evaluator.create_decision_curve_analysis(
            evaluation_results, condition_type,
            save_path=viz_dir / 'decision_curve_analysis.html'
        )
        
        print(f"Visualizations saved to: {viz_dir}")
    
    def setup_hybrid_system(self):
        """Setup hybrid inference system"""
        print("\n=== Setting up Hybrid Inference System ===")
        
        # Load trained models for all conditions
        all_models = {}
        for condition_type in DATASETS.keys():
            condition_models = self.ml_trainer.load_models(condition_type)
            for model_name, model in condition_models.items():
                all_models[f"{condition_type}_{model_name}"] = model
        
        # Initialize hybrid system
        self.hybrid_system = HybridInferenceSystem(
            self.config, all_models, self.knowledge_base
        )
        
        print("Hybrid inference system setup completed!")
    
    def predict_patient_risk(self, patient_data, condition_type):
        """Predict risk for a single patient"""
        if self.hybrid_system is None:
            self.setup_hybrid_system()
        
        # Make hybrid prediction
        result = self.hybrid_system.predict_risk_hybrid(patient_data, condition_type)
        
        return result
    
    def create_voice_interaction(self, condition_type, patient_symptoms=None, 
                               risk_assessment=None):
        """Create voice interaction for patient"""
        return self.voice_assistant.create_voice_interaction(
            condition_type, patient_symptoms, risk_assessment
        )
    
    def run_dashboard(self, port=8050):
        """Run the clinician dashboard"""
        dashboard = ClinicianDashboard(self.config)
        dashboard.run_dashboard(port=port)
    
    def process_all_datasets(self):
        """Process all available datasets"""
        print("\n=== Processing All Datasets ===")
        
        results = {}
        for condition_type in DATASETS.keys():
            try:
                result = self.process_dataset(condition_type)
                if result:
                    results[condition_type] = result
                    print(f"✓ {condition_type} dataset processed successfully")
                else:
                    print(f"✗ Failed to process {condition_type} dataset")
            except Exception as e:
                print(f"✗ Error processing {condition_type} dataset: {e}")
        
        return results
    
    def create_sample_patient_scenarios(self):
        """Create sample patient scenarios for demonstration"""
        print("\n=== Creating Sample Patient Scenarios ===")
        
        scenarios = {
            'kidney': {
                'patient_id': 'P001',
                'Age': 72,
                'Gender': 1,  # Male
                'BMI': 28.5,
                'SystolicBP': 145,
                'DiastolicBP': 95,
                'SerumCreatinine': 2.8,
                'eGFR': 25,
                'ProteinInUrine': 450,
                'ACR': 350,
                'CholesterolTotal': 220,
                'CholesterolLDL': 140,
                'CholesterolHDL': 35,
                'Hypertension_Flag': 1,
                'Diabetes_Flag': 0,
                'Obesity_Flag': 0
            },
            'diabetes': {
                'patient_id': 'P002',
                'age': 45,
                'gender': 'Female',
                'bmi': 32.1,
                'hypertension': 1,
                'heart_disease': 0,
                'smoking_history': 'former',
                'HbA1c_level': 7.8,
                'blood_glucose_level': 180,
                'family_history': 1
            },
            'heart': {
                'patient_id': 'P003',
                'Age': 58,
                'Sex': 1,  # Male
                'Chest pain type': 3,
                'BP': 160,
                'Cholesterol': 280,
                'FBS over 120': 1,
                'EKG results': 2,
                'Max HR': 140,
                'Exercise angina': 1,
                'ST depression': 1.8,
                'Slope of ST': 2,
                'Number of vessels fluro': 2,
                'Thallium': 6
            },
            'liver': {
                'patient_id': 'P004',
                'Age of the patient': 55,
                'Gender of the patient': 'Male',
                'Total Bilirubin': 2.5,
                'Direct Bilirubin': 1.2,
                'Alkphos Alkaline Phosphotase': 450,
                'Sgpt Alamine Aminotransferase': 85,
                'Sgot Aspartate Aminotransferase': 120,
                'Total Protiens': 6.2,
                'ALB Albumin': 2.8,
                'A/G Ratio Albumin and Globulin Ratio': 0.8
            }
        }
        
        # Save scenarios
        scenarios_path = self.config['RESULTS_DIR'] / 'sample_patient_scenarios.json'
        import json
        with open(scenarios_path, 'w') as f:
            json.dump(scenarios, f, indent=2)
        
        print(f"Sample scenarios saved to: {scenarios_path}")
        return scenarios
    
    def run_demo(self):
        """Run complete demonstration"""
        print("\n" + "="*60)
        print("AI-POWERED PROACTIVE PATIENT RISK ADVISOR - DEMO")
        print("="*60)
        
        # Setup knowledge base
        self.setup_knowledge_base()
        
        # Process all datasets
        results = self.process_all_datasets()
        
        # Setup hybrid system
        self.setup_hybrid_system()
        
        # Create sample scenarios
        scenarios = self.create_sample_patient_scenarios()
        
        # Demonstrate predictions
        print("\n=== DEMONSTRATION: Patient Risk Predictions ===")
        
        for condition_type, patient_data in scenarios.items():
            print(f"\n--- {condition_type.upper()} PATIENT ---")
            print(f"Patient ID: {patient_data['patient_id']}")
            
            # Make prediction
            prediction = self.predict_patient_risk(patient_data, condition_type)
            
            print(f"Risk Level: {prediction['final_risk_level']}")
            print(f"Risk Probability: {prediction['final_risk_probability']:.3f}")
            print(f"Confidence Score: {prediction['confidence_score']:.3f}")
            print(f"Key Risk Factors: {len(prediction['kbs_results']['matched_rules'])} rules matched")
            
            # Create voice interaction
            voice_interaction = self.create_voice_interaction(
                condition_type, 
                f"Patient reports symptoms for {condition_type}",
                prediction
            )
            
            print(f"Voice interaction created: {voice_interaction['initial_audio_path']}")
        
        print("\n=== DEMO COMPLETED ===")
        print("To run the clinician dashboard, use: system.run_dashboard()")
        print("To process specific datasets, use: system.process_dataset('condition_type')")
        
        return results

def main():
    """Main function to run the system"""
    print("Starting AI-Powered Proactive Patient Risk Advisor...")
    
    # Initialize system
    system = AIHealthcareSystem()
    
    # Run demo
    results = system.run_demo()
    
    # Optionally run dashboard
    # system.run_dashboard(port=8050)
    
    return system, results

if __name__ == "__main__":
    system, results = main()

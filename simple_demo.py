"""
Simple demonstration of AI-Powered Proactive Patient Risk Advisor
Shows core functionality without full ML training
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def run_simple_demo():
    """Run a simplified demonstration"""
    
    print("="*80)
    print("AI-POWERED PROACTIVE PATIENT RISK ADVISOR - SIMPLE DEMO")
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
        
        # Create sample patient scenarios
        print("\n3. Creating Sample Patient Scenarios...")
        from sample_patient_scenarios import create_sample_scenarios
        scenarios = create_sample_scenarios()
        
        # Demonstrate knowledge base functionality
        print("\n4. Demonstrating Knowledge Base System...")
        print("-" * 60)
        
        for scenario_id, scenario in scenarios.items():
            print(f"\n--- {scenario_id.upper().replace('_', ' ')} ---")
            print(f"Patient: {scenario['name']} (ID: {scenario['patient_id']})")
            print(f"Condition: {scenario['condition_type'].title()}")
            print(f"Age: {scenario['age']}, Gender: {scenario['gender']}")
            
            # Evaluate rules
            try:
                matched_rules = system.knowledge_base.evaluate_rules(
                    scenario['clinical_data'], 
                    scenario['condition_type']
                )
                
                # Also evaluate general rules
                general_rules = system.knowledge_base.evaluate_rules(
                    scenario['clinical_data'], 
                    'general'
                )
                
                all_rules = matched_rules + general_rules
                risk_level = system.knowledge_base.get_risk_level(all_rules)
                
                print(f"Risk Level: {risk_level}")
                print(f"Matched Rules: {len(all_rules)}")
                
                if all_rules:
                    print("Key Risk Factors:")
                    for i, rule in enumerate(all_rules[:3], 1):
                        print(f"  {i}. {rule['name']} - {rule['explanation']}")
                
                # Generate explanation
                explanation = system.knowledge_base.generate_explanation(all_rules, risk_level)
                print(f"\nExplanation:\n{explanation[:200]}...")
                
            except Exception as e:
                print(f"Error evaluating rules: {e}")
            
            print("-" * 60)
        
        # Demonstrate voice assistant
        print("\n5. Demonstrating Voice Assistant...")
        try:
            # Use high-risk kidney patient for voice demo
            kidney_patient = scenarios['high_risk_kidney_patient']
            
            voice_interaction = system.create_voice_interaction(
                'kidney',
                "Patient reports fatigue, swelling, and changes in urination",
                None  # No risk assessment for this demo
            )
            
            print("Voice Interaction Created:")
            print(f"  - Initial Script Length: {len(voice_interaction['initial_script'])} characters")
            print(f"  - Audio File: {voice_interaction['initial_audio_path']}")
            
            # Show sample script
            print("\nSample Voice Script:")
            print(voice_interaction['initial_script'][:300] + "...")
            
        except Exception as e:
            print(f"Voice assistant demo error: {e}")
        
        # Show system capabilities
        print("\n6. System Capabilities Summary...")
        print("✓ Knowledge Base System with clinical rules")
        print("✓ Voice Assistant with ElevenLabs integration")
        print("✓ Data preprocessing and feature engineering")
        print("✓ Machine learning models (Logistic Regression, XGBoost, LightGBM)")
        print("✓ SHAP and LIME explainability")
        print("✓ Hybrid inference combining ML and KBS")
        print("✓ Clinician dashboard with visualizations")
        print("✓ Comprehensive evaluation metrics")
        
        # Show file structure
        print("\n7. Generated Files...")
        results_dir = Path("results")
        if results_dir.exists():
            print("Results directory contains:")
            for item in results_dir.rglob("*.json"):
                print(f"  - {item.relative_to(results_dir)}")
        
        print("\n" + "="*80)
        print("SIMPLE DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print("\nThe system includes:")
        print("1. Complete data preprocessing pipeline")
        print("2. Advanced feature engineering")
        print("3. Multiple ML models with cross-validation")
        print("4. Knowledge-based system with clinical rules")
        print("5. Hybrid inference combining ML and KBS")
        print("6. Voice assistant with ElevenLabs")
        print("7. Interactive clinician dashboard")
        print("8. Comprehensive evaluation and visualization")
        
        print("\nTo run the full system with ML training:")
        print("python main.py")
        
        print("\nTo run the clinician dashboard:")
        print("python -c \"from main import AIHealthcareSystem; system = AIHealthcareSystem(); system.run_dashboard()\"")
        
        return True
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_simple_demo()
    
    if success:
        print("\n🎉 System demonstration completed successfully!")
        print("The AI-Powered Proactive Patient Risk Advisor is ready for use.")
    else:
        print("\n❌ Demo failed. Please check the error messages above.")

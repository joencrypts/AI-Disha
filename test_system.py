"""
Test script for AI-Powered Proactive Patient Risk Advisor
Verifies system components work correctly
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing module imports...")
    
    try:
        import config
        print("✓ config.py imported successfully")
    except Exception as e:
        print(f"✗ config.py import failed: {e}")
        return False
    
    try:
        from data_preprocessing import DataPreprocessor
        print("✓ data_preprocessing.py imported successfully")
    except Exception as e:
        print(f"✗ data_preprocessing.py import failed: {e}")
        return False
    
    try:
        from feature_engineering import FeatureEngineer
        print("✓ feature_engineering.py imported successfully")
    except Exception as e:
        print(f"✗ feature_engineering.py import failed: {e}")
        return False
    
    try:
        from ml_models import MLModelTrainer
        print("✓ ml_models.py imported successfully")
    except Exception as e:
        print(f"✗ ml_models.py import failed: {e}")
        return False
    
    try:
        from knowledge_base import KnowledgeBaseSystem
        print("✓ knowledge_base.py imported successfully")
    except Exception as e:
        print(f"✗ knowledge_base.py import failed: {e}")
        return False
    
    try:
        from hybrid_inference import HybridInferenceSystem
        print("✓ hybrid_inference.py imported successfully")
    except Exception as e:
        print(f"✗ hybrid_inference.py import failed: {e}")
        return False
    
    try:
        from voice_assistant import VoiceAssistant
        print("✓ voice_assistant.py imported successfully")
    except Exception as e:
        print(f"✗ voice_assistant.py import failed: {e}")
        return False
    
    try:
        from evaluation import ModelEvaluator
        print("✓ evaluation.py imported successfully")
    except Exception as e:
        print(f"✗ evaluation.py import failed: {e}")
        return False
    
    return True

def test_data_loading():
    """Test data loading functionality"""
    print("\nTesting data loading...")
    
    try:
        import config
        from data_preprocessing import DataPreprocessor
        
        preprocessor = DataPreprocessor({
            'DATA_DIR': config.DATA_DIR,
            'DATASETS': config.DATASETS
        })
        
        # Test loading each dataset
        for condition_type in config.DATASETS.keys():
            df = preprocessor.load_dataset(condition_type)
            if df is not None:
                print(f"✓ {condition_type} dataset loaded: {df.shape}")
            else:
                print(f"✗ {condition_type} dataset failed to load")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        return False

def test_knowledge_base():
    """Test knowledge base functionality"""
    print("\nTesting knowledge base...")
    
    try:
        import config
        from knowledge_base import KnowledgeBaseSystem
        
        kb = KnowledgeBaseSystem({
            'CLINICAL_THRESHOLDS': config.CLINICAL_THRESHOLDS,
            'RISK_LEVELS': config.RISK_LEVELS
        })
        
        # Create rules
        rules = kb.create_clinical_rules()
        
        if rules and len(rules) > 0:
            print(f"✓ Knowledge base created with {len(rules)} condition types")
            
            # Test rule evaluation
            test_patient = {
                'eGFR': 25,
                'SystolicBP': 145,
                'Age': 72
            }
            
            matched_rules = kb.evaluate_rules(test_patient, 'kidney')
            print(f"✓ Rule evaluation works: {len(matched_rules)} rules matched")
            
            return True
        else:
            print("✗ Knowledge base creation failed")
            return False
            
    except Exception as e:
        print(f"✗ Knowledge base test failed: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering functionality"""
    print("\nTesting feature engineering...")
    
    try:
        import config
        from feature_engineering import FeatureEngineer
        
        fe = FeatureEngineer({
            'FEATURE_CONFIG': config.FEATURE_CONFIG,
            'CLINICAL_THRESHOLDS': config.CLINICAL_THRESHOLDS
        })
        
        # Create sample data
        sample_data = pd.DataFrame({
            'Age': [65, 45, 72],
            'BMI': [28.5, 32.1, 25.2],
            'SystolicBP': [145, 130, 120],
            'DiastolicBP': [95, 85, 80],
            'SerumCreatinine': [2.1, 1.2, 0.9],
            'Gender': [1, 0, 1]
        })
        
        # Test feature engineering
        engineered_data = fe.engineer_features(sample_data, 'kidney')
        
        if engineered_data.shape[1] > sample_data.shape[1]:
            print(f"✓ Feature engineering works: {sample_data.shape[1]} -> {engineered_data.shape[1]} features")
            return True
        else:
            print("✗ Feature engineering failed to create new features")
            return False
            
    except Exception as e:
        print(f"✗ Feature engineering test failed: {e}")
        return False

def test_voice_assistant():
    """Test voice assistant functionality"""
    print("\nTesting voice assistant...")
    
    try:
        import config
        from voice_assistant import VoiceAssistant
        
        va = VoiceAssistant({
            'ELEVENLABS_CONFIG': config.ELEVENLABS_CONFIG,
            'STATIC_DIR': config.STATIC_DIR
        })
        
        # Test script generation
        script = va.create_patient_interaction_script('kidney')
        
        if script and len(script) > 100:
            print(f"✓ Voice assistant script generation works: {len(script)} characters")
            return True
        else:
            print("✗ Voice assistant script generation failed")
            return False
            
    except Exception as e:
        print(f"✗ Voice assistant test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("AI-POWERED PROACTIVE PATIENT RISK ADVISOR - SYSTEM TESTS")
    print("="*60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Data Loading", test_data_loading),
        ("Knowledge Base", test_knowledge_base),
        ("Feature Engineering", test_feature_engineering),
        ("Voice Assistant", test_voice_assistant)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                print(f"✓ {test_name} PASSED")
                passed += 1
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\nTo run the full demonstration:")
        print("python demo.py")
        print("\nTo run the main system:")
        print("python main.py")
    else:
        print("\nPlease fix the failing tests before running the system.")

"""
Simplified model training script for AI-Powered Proactive Patient Risk Advisor
Trains ML models for all conditions and saves them for web interface
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import core modules
from config import *
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from ml_models import MLModelTrainer
from knowledge_base import KnowledgeBaseSystem
from dataset_processor import DatasetProcessor

def train_all_models():
    """Train ML models for all conditions"""
    print("="*60)
    print("TRAINING ML MODELS FOR ALL CONDITIONS")
    print("="*60)
    
    # Initialize components
    config = {
        'DATA_DIR': DATA_DIR,
        'MODELS_DIR': MODELS_DIR,
        'RESULTS_DIR': RESULTS_DIR,
        'DATASETS': DATASETS,
        'MODEL_CONFIG': MODEL_CONFIG,
        'FEATURE_CONFIG': FEATURE_CONFIG,
        'CLINICAL_THRESHOLDS': CLINICAL_THRESHOLDS,
        'RISK_LEVELS': RISK_LEVELS
    }
    
    preprocessor = DataPreprocessor(config)
    feature_engineer = FeatureEngineer(config)
    ml_trainer = MLModelTrainer(config)
    knowledge_base = KnowledgeBaseSystem(config)
    dataset_processor = DatasetProcessor(config)
    
    # Setup knowledge base
    print("\n=== Setting up Knowledge Base ===")
    knowledge_base.create_clinical_rules()
    rules_path = config['RESULTS_DIR'] / 'clinical_rules.json'
    knowledge_base.save_rules_to_json(rules_path)
    print("Knowledge base setup completed!")
    
    # Train models for each condition
    results = {}
    for condition_type in DATASETS.keys():
        print(f"\n=== Training models for {condition_type} ===")
        
        try:
            # Load dataset
            dataset_config = DATASETS[condition_type]
            file_path = DATA_DIR / dataset_config['file']
            
            print(f"Loading {condition_type} dataset from {file_path}")
            
            # Handle encoding issues for liver dataset
            if condition_type == 'liver':
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='latin-1')
            else:
                df = pd.read_csv(file_path)
            
            print(f"Original dataset shape: {df.shape}")
            
            # Process dataset with specific handling
            df_processed = dataset_processor.process_dataset(condition_type, df)
            
            # Standardize features
            df_processed = dataset_processor.standardize_features(df_processed, condition_type)
            
            # Feature engineering
            df_engineered = feature_engineer.engineer_features(
                df_processed, 
                condition_type
            )
            
            # Train ML models
            target_col = 'target'  # Use standardized target column
            models_results, X_test, y_test, feature_names = ml_trainer.train_all_models(
                df_engineered, 
                target_col, 
                condition_type
            )
            
            # Save models
            ml_trainer.save_models(condition_type, models_results)
            
            results[condition_type] = {
                'data': df_engineered,
                'models': models_results,
                'feature_names': feature_names
            }
            
            print(f"✓ {condition_type} models trained and saved successfully")
            
        except Exception as e:
            print(f"✗ Error training {condition_type} models: {e}")
            continue
    
    print(f"\n=== Training Complete ===")
    print(f"Models saved to: {MODELS_DIR}")
    print(f"Knowledge base saved to: {rules_path}")
    
    return results

if __name__ == "__main__":
    results = train_all_models()
    print("\nAll models trained successfully!")

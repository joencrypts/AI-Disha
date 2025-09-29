"""
Configuration file for AI-Powered Proactive Patient Risk Advisor
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
STATIC_DIR = PROJECT_ROOT / "static"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, STATIC_DIR]:
    directory.mkdir(exist_ok=True)

# Dataset configurations
DATASETS = {
    'kidney': {
        'file': 'kidney_disease.csv',
        'target': 'classification',
        'positive_class': 'ckd',
        'columns_mapping': {
            'age': 'Age',
            'bp': 'SystolicBP',
            'bgr': 'BloodGlucose',
            'bu': 'BloodUrea',
            'sc': 'SerumCreatinine',
            'sod': 'Sodium',
            'pot': 'Potassium',
            'hemo': 'Hemoglobin',
            'pcv': 'PackedCellVolume',
            'wc': 'WhiteBloodCells',
            'rc': 'RedBloodCells',
            'htn': 'Hypertension',
            'dm': 'Diabetes',
            'cad': 'CoronaryArteryDisease'
        }
    },
    'diabetes': {
        'file': 'diabetes_prediction_dataset.csv',
        'target': 'diabetes',
        'positive_class': 1,
        'columns_mapping': {
            'age': 'Age',
            'gender': 'Gender',
            'bmi': 'BMI',
            'hypertension': 'Hypertension',
            'heart_disease': 'HeartDisease',
            'smoking_history': 'SmokingHistory',
            'HbA1c_level': 'HbA1c',
            'blood_glucose_level': 'BloodGlucose'
        }
    },
    'heart': {
        'file': 'Heart_Disease_Prediction.csv',
        'target': 'Heart Disease',
        'positive_class': 'Presence',
        'columns_mapping': {
            'Age': 'Age',
            'Sex': 'Gender',
            'BP': 'SystolicBP',
            'Cholesterol': 'Cholesterol',
            'FBS over 120': 'FastingBloodSugar',
            'Max HR': 'MaxHeartRate',
            'Exercise angina': 'ExerciseAngina',
            'ST depression': 'STDepression'
        }
    },
    'liver': {
        'file': 'Liver Patient Dataset (LPD)_train.csv',
        'target': 'Result',
        'positive_class': 1,
        'columns_mapping': {
            'Age of the patient': 'Age',
            'Gender of the patient': 'Gender',
            'Total Bilirubin': 'TotalBilirubin',
            'Direct Bilirubin': 'DirectBilirubin',
            'Alkphos Alkaline Phosphotase': 'AlkalinePhosphatase',
            'Sgpt Alamine Aminotransferase': 'ALT',
            'Sgot Aspartate Aminotransferase': 'AST',
            'Total Protiens': 'TotalProteins',
            'ALB Albumin': 'Albumin',
            'A/G Ratio Albumin and Globulin Ratio': 'AlbuminGlobulinRatio'
        }
    }
}

# Model configurations
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'stratify': True
}

# Feature engineering configurations
FEATURE_CONFIG = {
    'bmi_categories': {
        'underweight': (0, 18.5),
        'normal': (18.5, 25),
        'overweight': (25, 30),
        'obese': (30, float('inf'))
    },
    'bp_categories': {
        'normal': (0, 120),
        'elevated': (120, 130),
        'high_stage1': (130, 140),
        'high_stage2': (140, float('inf'))
    },
    'age_groups': {
        'young': (0, 30),
        'middle': (30, 60),
        'senior': (60, float('inf'))
    }
}

# Risk levels
RISK_LEVELS = {
    'LOW': 0,
    'MODERATE': 1,
    'HIGH': 2
}

# ElevenLabs configuration
ELEVENLABS_CONFIG = {
    'api_key': os.getenv('ELEVENLABS_API_KEY', ''),
    'voice_id': 'pNInz6obpgDQGcFmaJgB',  # Default voice
    'model_id': 'eleven_monolingual_v1'
}

# Clinical guidelines thresholds
CLINICAL_THRESHOLDS = {
    'kidney': {
        'egfr_normal': 90,
        'egfr_mild': 60,
        'egfr_moderate': 30,
        'creatinine_high': 1.2,
        'acr_high': 30
    },
    'diabetes': {
        'hba1c_normal': 5.7,
        'hba1c_prediabetes': 6.4,
        'glucose_normal': 100,
        'glucose_prediabetes': 126
    },
    'heart': {
        'cholesterol_high': 200,
        'ldl_high': 100,
        'hdl_low': 40,
        'bp_high': 140
    },
    'liver': {
        'alt_high': 56,
        'ast_high': 40,
        'bilirubin_high': 1.2,
        'albumin_low': 3.5
    }
}

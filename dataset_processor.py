"""
Dataset-specific processing for different medical datasets
Handles column mapping, data cleaning, and standardization
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DatasetProcessor:
    """Process different medical datasets with specific handling"""
    
    def __init__(self, config):
        self.config = config
        self.encoders = {}
        self.scalers = {}
        self.imputers = {}
        
    def process_kidney_dataset(self, df):
        """Process kidney disease dataset"""
        print("Processing Kidney Disease dataset...")
        
        # Handle missing values and data types
        df = df.copy()
        
        # Convert numeric columns
        numeric_cols = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle categorical columns
        categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace('', np.nan)
        
        # Encode categorical variables
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    # Handle missing values before encoding
                    df[col] = df[col].fillna('unknown')
                    df[col] = self.encoders[col].fit_transform(df[col])
                else:
                    df[col] = df[col].fillna('unknown')
                    # Handle unseen categories
                    unique_values = set(df[col].unique())
                    known_values = set(self.encoders[col].classes_)
                    new_values = unique_values - known_values
                    if new_values:
                        # Add new categories
                        all_values = list(known_values) + list(new_values)
                        self.encoders[col] = LabelEncoder()
                        self.encoders[col].fit(all_values)
                    df[col] = self.encoders[col].transform(df[col])
        
        # Handle missing values in numeric columns
        for col in numeric_cols:
            if col in df.columns:
                if col not in self.imputers:
                    self.imputers[col] = SimpleImputer(strategy='median')
                    df[col] = self.imputers[col].fit_transform(df[[col]]).flatten()
                else:
                    df[col] = self.imputers[col].transform(df[[col]]).flatten()
        
        # Final cleanup - ensure no NaN values remain
        df = df.fillna(0)
        
        # Create target variable (binary classification)
        df['target'] = (df['classification'] == 'ckd').astype(int)
        
        # Drop original classification column
        df = df.drop('classification', axis=1)
        
        print(f"Kidney dataset processed: {df.shape}")
        return df
    
    def process_diabetes_dataset(self, df):
        """Process diabetes prediction dataset"""
        print("Processing Diabetes dataset...")
        
        df = df.copy()
        
        # Handle gender encoding
        if 'gender' in df.columns:
            df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
        
        # Handle smoking history encoding
        if 'smoking_history' in df.columns:
            smoking_mapping = {
                'never': 0,
                'No Info': 1,
                'former': 2,
                'current': 3,
                'not current': 2
            }
            df['smoking_history'] = df['smoking_history'].map(smoking_mapping)
            df['smoking_history'] = df['smoking_history'].fillna(1)  # Fill unknown with 'No Info'
        
        # Handle missing values
        numeric_cols = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median())
        
        # Create target variable
        df['target'] = df['diabetes'].astype(int)
        df = df.drop('diabetes', axis=1)
        
        # Final cleanup
        df = df.fillna(0)
        
        print(f"Diabetes dataset processed: {df.shape}")
        return df
    
    def process_heart_dataset(self, df):
        """Process heart disease dataset"""
        print("Processing Heart Disease dataset...")
        
        df = df.copy()
        
        # Handle gender encoding (already numeric)
        if 'Sex' in df.columns:
            df['Sex'] = df['Sex'].astype(int)
        
        # Handle categorical variables
        categorical_cols = ['Chest pain type', 'EKG results', 'Slope of ST', 'Thallium']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median())
        
        # Handle missing values in numeric columns
        numeric_cols = ['Age', 'BP', 'Cholesterol', 'FBS over 120', 'Max HR', 'Exercise angina', 'ST depression', 'Number of vessels fluro']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median())
        
        # Create target variable
        df['target'] = (df['Heart Disease'] == 'Presence').astype(int)
        df = df.drop('Heart Disease', axis=1)
        
        print(f"Heart dataset processed: {df.shape}")
        return df
    
    def process_liver_dataset(self, df):
        """Process liver disease dataset"""
        print("Processing Liver Disease dataset...")
        
        df = df.copy()
        
        # Handle gender encoding
        if 'Gender of the patient' in df.columns:
            df['Gender of the patient'] = df['Gender of the patient'].map({'Male': 1, 'Female': 0})
        
        # Handle missing values and data types
        numeric_cols = ['Age of the patient', 'Total Bilirubin', 'Direct Bilirubin', 
                       'Alkphos Alkaline Phosphotase', 'Sgpt Alamine Aminotransferase',
                       'Sgot Aspartate Aminotransferase', 'Total Protiens', 'ALB Albumin',
                       'A/G Ratio Albumin and Globulin Ratio']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median())
        
        # Create target variable (binary classification)
        df['target'] = (df['Result'] == 1).astype(int)  # Convert to binary: 1 = disease, 0 = no disease
        df = df.drop('Result', axis=1)
        
        # Final cleanup
        df = df.fillna(0)
        
        print(f"Liver dataset processed: {df.shape}")
        return df
    
    def process_dataset(self, condition_type, df):
        """Process dataset based on condition type"""
        if condition_type == 'kidney':
            return self.process_kidney_dataset(df)
        elif condition_type == 'diabetes':
            return self.process_diabetes_dataset(df)
        elif condition_type == 'heart':
            return self.process_heart_dataset(df)
        elif condition_type == 'liver':
            return self.process_liver_dataset(df)
        else:
            raise ValueError(f"Unknown condition type: {condition_type}")
    
    def standardize_features(self, df, condition_type):
        """Standardize features for ML models"""
        print(f"Standardizing features for {condition_type}...")
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'target']
        
        # Scale features
        if condition_type not in self.scalers:
            self.scalers[condition_type] = StandardScaler()
            df[numeric_cols] = self.scalers[condition_type].fit_transform(df[numeric_cols])
        else:
            df[numeric_cols] = self.scalers[condition_type].transform(df[numeric_cols])
        
        return df
    
    def get_feature_importance_mapping(self, condition_type):
        """Get feature importance mapping for explainability"""
        mappings = {
            'kidney': {
                'age': 'Age',
                'bp': 'Blood Pressure',
                'bgr': 'Blood Glucose',
                'bu': 'Blood Urea',
                'sc': 'Serum Creatinine',
                'hemo': 'Hemoglobin',
                'htn': 'Hypertension',
                'dm': 'Diabetes'
            },
            'diabetes': {
                'age': 'Age',
                'bmi': 'BMI',
                'HbA1c_level': 'HbA1c Level',
                'blood_glucose_level': 'Blood Glucose',
                'hypertension': 'Hypertension',
                'heart_disease': 'Heart Disease'
            },
            'heart': {
                'Age': 'Age',
                'BP': 'Blood Pressure',
                'Cholesterol': 'Cholesterol',
                'Max HR': 'Max Heart Rate',
                'Exercise angina': 'Exercise Angina'
            },
            'liver': {
                'Age of the patient': 'Age',
                'Total Bilirubin': 'Total Bilirubin',
                'ALB Albumin': 'Albumin',
                'Sgpt Alamine Aminotransferase': 'ALT',
                'Sgot Aspartate Aminotransferase': 'AST'
            }
        }
        return mappings.get(condition_type, {})

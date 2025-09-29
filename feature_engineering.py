"""
Feature engineering module for AI-Powered Proactive Patient Risk Advisor
Creates clinical features like BMI categories, eGFR trends, ratios, and risk scores
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Advanced feature engineering for clinical data"""
    
    def __init__(self, config):
        self.config = config
        self.feature_config = config['FEATURE_CONFIG']
        self.clinical_thresholds = config['CLINICAL_THRESHOLDS']
        
    def create_bmi_categories(self, df, bmi_col='BMI'):
        """Create BMI categories based on WHO standards"""
        if bmi_col not in df.columns:
            print(f"BMI column '{bmi_col}' not found")
            return df
        
        df = df.copy()
        df['BMI_Category'] = pd.cut(
            df[bmi_col], 
            bins=[0, 18.5, 25, 30, float('inf')],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese'],
            include_lowest=True
        )
        
        # One-hot encode BMI categories
        bmi_dummies = pd.get_dummies(df['BMI_Category'], prefix='BMI')
        df = pd.concat([df, bmi_dummies], axis=1)
        df = df.drop('BMI_Category', axis=1)
        
        return df
    
    def create_bp_categories(self, df, systolic_col='SystolicBP', diastolic_col='DiastolicBP'):
        """Create blood pressure categories based on AHA guidelines"""
        df = df.copy()
        
        if systolic_col in df.columns and diastolic_col in df.columns:
            # Create BP categories based on systolic pressure
            df['BP_Category'] = pd.cut(
                df[systolic_col],
                bins=[0, 120, 130, 140, float('inf')],
                labels=['Normal', 'Elevated', 'High_Stage1', 'High_Stage2'],
                include_lowest=True
            )
            
            # One-hot encode BP categories
            bp_dummies = pd.get_dummies(df['BP_Category'], prefix='BP')
            df = pd.concat([df, bp_dummies], axis=1)
            df = df.drop('BP_Category', axis=1)
            
            # Calculate pulse pressure
            df['Pulse_Pressure'] = df[systolic_col] - df[diastolic_col]
            
        return df
    
    def create_age_groups(self, df, age_col='Age'):
        """Create age groups for better risk stratification"""
        if age_col not in df.columns:
            print(f"Age column '{age_col}' not found")
            return df
        
        df = df.copy()
        df['Age_Group'] = pd.cut(
            df[age_col],
            bins=[0, 30, 60, float('inf')],
            labels=['Young', 'Middle', 'Senior'],
            include_lowest=True
        )
        
        # One-hot encode age groups
        age_dummies = pd.get_dummies(df['Age_Group'], prefix='Age')
        df = pd.concat([df, age_dummies], axis=1)
        df = df.drop('Age_Group', axis=1)
        
        return df
    
    def calculate_egfr_trend(self, df, creatinine_col='SerumCreatinine', age_col='Age', gender_col='Gender'):
        """Calculate eGFR and trend indicators"""
        df = df.copy()
        
        if creatinine_col in df.columns and age_col in df.columns:
            # Calculate eGFR using MDRD formula
            # eGFR = 175 × (Scr)^-1.154 × (Age)^-0.203 × (0.742 if female)
            gender_factor = 0.742 if gender_col in df.columns else 1.0
            if gender_col in df.columns:
                gender_factor = np.where(df[gender_col] == 0, 0.742, 1.0)  # Assuming 0 = Female
            
            df['eGFR'] = 175 * (df[creatinine_col] ** -1.154) * (df[age_col] ** -0.203) * gender_factor
            
            # Create eGFR categories
            df['eGFR_Category'] = pd.cut(
                df['eGFR'],
                bins=[0, 30, 60, 90, float('inf')],
                labels=['Severe', 'Moderate', 'Mild', 'Normal'],
                include_lowest=True
            )
            
            # One-hot encode eGFR categories
            egfr_dummies = pd.get_dummies(df['eGFR_Category'], prefix='eGFR')
            df = pd.concat([df, egfr_dummies], axis=1)
            df = df.drop('eGFR_Category', axis=1)
        
        return df
    
    def calculate_liver_ratios(self, df, alt_col='Sgpt Alamine Aminotransferase', 
                              ast_col='Sgot Aspartate Aminotransferase',
                              bilirubin_col='Total Bilirubin', albumin_col='ALB Albumin'):
        """Calculate liver function ratios and indicators"""
        df = df.copy()
        
        # AST/ALT ratio
        if ast_col in df.columns and alt_col in df.columns:
            df['AST_ALT_Ratio'] = df[ast_col] / (df[alt_col] + 1e-8)  # Add small value to avoid division by zero
            
        # Bilirubin/Albumin ratio
        if bilirubin_col in df.columns and albumin_col in df.columns:
            df['Bilirubin_Albumin_Ratio'] = df[bilirubin_col] / (df[albumin_col] + 1e-8)
        
        # Liver function score
        liver_score = 0
        if ast_col in df.columns:
            liver_score += (df[ast_col] > self.clinical_thresholds['liver']['ast_high']).astype(int)
        if alt_col in df.columns:
            liver_score += (df[alt_col] > self.clinical_thresholds['liver']['alt_high']).astype(int)
        if bilirubin_col in df.columns:
            liver_score += (df[bilirubin_col] > self.clinical_thresholds['liver']['bilirubin_high']).astype(int)
        if albumin_col in df.columns:
            liver_score += (df[albumin_col] < self.clinical_thresholds['liver']['albumin_low']).astype(int)
        
        df['Liver_Function_Score'] = liver_score
        
        return df
    
    def calculate_cholesterol_ratios(self, df, ldl_col='CholesterolLDL', hdl_col='CholesterolHDL', 
                                   total_col='CholesterolTotal'):
        """Calculate cholesterol ratios and cardiovascular risk indicators"""
        df = df.copy()
        
        # LDL/HDL ratio
        if ldl_col in df.columns and hdl_col in df.columns:
            df['LDL_HDL_Ratio'] = df[ldl_col] / (df[hdl_col] + 1e-8)
        
        # Total/HDL ratio
        if total_col in df.columns and hdl_col in df.columns:
            df['Total_HDL_Ratio'] = df[total_col] / (df[hdl_col] + 1e-8)
        
        # Cardiovascular risk score
        cv_score = 0
        if ldl_col in df.columns:
            cv_score += (df[ldl_col] > self.clinical_thresholds['heart']['ldl_high']).astype(int)
        if hdl_col in df.columns:
            cv_score += (df[hdl_col] < self.clinical_thresholds['heart']['hdl_low']).astype(int)
        if total_col in df.columns:
            cv_score += (df[total_col] > self.clinical_thresholds['heart']['cholesterol_high']).astype(int)
        
        df['Cardiovascular_Risk_Score'] = cv_score
        
        return df
    
    def create_comorbidity_flags(self, df):
        """Create comorbidity flags based on available conditions"""
        df = df.copy()
        
        # Hypertension flag
        if 'SystolicBP' in df.columns:
            df['Hypertension_Flag'] = (df['SystolicBP'] > 140).astype(int)
        
        # Diabetes flag (based on glucose or HbA1c)
        if 'FastingBloodSugar' in df.columns:
            df['Diabetes_Flag'] = (df['FastingBloodSugar'] > 126).astype(int)
        elif 'blood_glucose_level' in df.columns:
            df['Diabetes_Flag'] = (df['blood_glucose_level'] > 126).astype(int)
        
        # Obesity flag
        if 'BMI' in df.columns:
            df['Obesity_Flag'] = (df['BMI'] > 30).astype(int)
        
        # Kidney disease flag (based on eGFR or creatinine)
        if 'eGFR' in df.columns:
            df['Kidney_Disease_Flag'] = (df['eGFR'] < 60).astype(int)
        elif 'SerumCreatinine' in df.columns:
            df['Kidney_Disease_Flag'] = (df['SerumCreatinine'] > 1.2).astype(int)
        
        return df
    
    def calculate_risk_scores(self, df, condition_type):
        """Calculate condition-specific risk scores"""
        df = df.copy()
        
        if condition_type == 'kidney':
            risk_score = 0
            
            # Age factor
            if 'Age' in df.columns:
                risk_score += np.where(df['Age'] > 65, 2, 0)
            
            # eGFR factor
            if 'eGFR' in df.columns:
                risk_score += np.where(df['eGFR'] < 30, 3, 
                                     np.where(df['eGFR'] < 60, 2, 
                                             np.where(df['eGFR'] < 90, 1, 0)))
            
            # Proteinuria factor
            if 'ProteinInUrine' in df.columns:
                risk_score += np.where(df['ProteinInUrine'] > 0.3, 2, 0)
            
            # Hypertension factor
            if 'SystolicBP' in df.columns:
                risk_score += np.where(df['SystolicBP'] > 140, 1, 0)
            
            df['Kidney_Risk_Score'] = risk_score
            
        elif condition_type == 'diabetes':
            risk_score = 0
            
            # Age factor
            if 'age' in df.columns:
                risk_score += np.where(df['age'] > 45, 1, 0)
            
            # BMI factor
            if 'bmi' in df.columns:
                risk_score += np.where(df['bmi'] > 30, 2, 
                                     np.where(df['bmi'] > 25, 1, 0))
            
            # Family history factor
            if 'family_history' in df.columns:
                risk_score += df['family_history']
            
            # Hypertension factor
            if 'hypertension' in df.columns:
                risk_score += df['hypertension']
            
            df['Diabetes_Risk_Score'] = risk_score
            
        elif condition_type == 'heart':
            risk_score = 0
            
            # Age factor
            if 'Age' in df.columns:
                risk_score += np.where(df['Age'] > 55, 2, 
                                     np.where(df['Age'] > 45, 1, 0))
            
            # Cholesterol factor
            if 'Cholesterol' in df.columns:
                risk_score += np.where(df['Cholesterol'] > 240, 2, 
                                     np.where(df['Cholesterol'] > 200, 1, 0))
            
            # Blood pressure factor
            if 'BP' in df.columns:
                risk_score += np.where(df['BP'] > 140, 2, 
                                     np.where(df['BP'] > 120, 1, 0))
            
            # Smoking factor
            if 'smoking' in df.columns:
                risk_score += df['smoking']
            
            df['Heart_Risk_Score'] = risk_score
            
        elif condition_type == 'liver':
            risk_score = 0
            
            # Age factor
            if 'Age of the patient' in df.columns:
                risk_score += np.where(df['Age of the patient'] > 50, 1, 0)
            
            # Gender factor (males at higher risk)
            if 'Gender of the patient' in df.columns:
                risk_score += np.where(df['Gender of the patient'] == 'Male', 1, 0)
            
            # Bilirubin factor
            if 'Total Bilirubin' in df.columns:
                risk_score += np.where(df['Total Bilirubin'] > 1.2, 2, 0)
            
            # Albumin factor
            if 'ALB Albumin' in df.columns:
                risk_score += np.where(df['ALB Albumin'] < 3.5, 2, 0)
            
            df['Liver_Risk_Score'] = risk_score
        
        return df
    
    def create_interaction_features(self, df):
        """Create interaction features between important variables"""
        df = df.copy()
        
        # Age-BMI interaction
        if 'Age' in df.columns and 'BMI' in df.columns:
            df['Age_BMI_Interaction'] = df['Age'] * df['BMI']
        
        # Age-BP interaction
        if 'Age' in df.columns and 'SystolicBP' in df.columns:
            df['Age_BP_Interaction'] = df['Age'] * df['SystolicBP']
        
        # BMI-BP interaction
        if 'BMI' in df.columns and 'SystolicBP' in df.columns:
            df['BMI_BP_Interaction'] = df['BMI'] * df['SystolicBP']
        
        return df
    
    def engineer_features(self, df, condition_type):
        """Complete feature engineering pipeline"""
        print(f"\n=== Feature Engineering for {condition_type} ===")
        
        df = df.copy()
        original_features = len(df.columns)
        
        # Basic demographic features
        df = self.create_age_groups(df)
        df = self.create_bmi_categories(df)
        df = self.create_bp_categories(df)
        
        # Condition-specific features
        if condition_type == 'kidney':
            df = self.calculate_egfr_trend(df)
            df = self.calculate_cholesterol_ratios(df)
        elif condition_type == 'diabetes':
            df = self.calculate_cholesterol_ratios(df)
        elif condition_type == 'heart':
            df = self.calculate_cholesterol_ratios(df)
        elif condition_type == 'liver':
            df = self.calculate_liver_ratios(df)
        
        # Common features
        df = self.create_comorbidity_flags(df)
        df = self.calculate_risk_scores(df, condition_type)
        df = self.create_interaction_features(df)
        
        new_features = len(df.columns) - original_features
        print(f"Created {new_features} new features")
        print(f"Total features: {len(df.columns)}")
        
        return df
    
    def get_feature_importance_ranking(self, df, target_col):
        """Get feature importance ranking for feature selection"""
        if target_col not in df.columns:
            return None
        
        # Calculate correlation with target
        correlations = df.corr()[target_col].abs().sort_values(ascending=False)
        
        # Remove target column from correlations
        correlations = correlations.drop(target_col, errors='ignore')
        
        return correlations

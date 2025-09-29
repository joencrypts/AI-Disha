"""
Data preprocessing module for AI-Powered Proactive Patient Risk Advisor
Handles missing values, outliers, normalization, and encoding
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Comprehensive data preprocessing pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.outlier_detectors = {}
        
    def load_dataset(self, dataset_name):
        """Load dataset based on configuration"""
        dataset_config = self.config['DATASETS'][dataset_name]
        file_path = self.config['DATA_DIR'] / dataset_config['file']
        
        try:
            # Try different encodings for problematic files
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"Loaded {dataset_name} dataset with {encoding} encoding: {df.shape}")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                print(f"Error loading {dataset_name} dataset: Could not decode with any encoding")
                return None
                
            return df
        except Exception as e:
            print(f"Error loading {dataset_name} dataset: {e}")
            return None
    
    def handle_missing_values(self, df, strategy='auto'):
        """Handle missing values using various strategies"""
        print(f"Handling missing values. Strategy: {strategy}")
        
        # Check missing values
        missing_info = df.isnull().sum()
        missing_cols = missing_info[missing_info > 0]
        
        if len(missing_cols) == 0:
            print("No missing values found")
            return df
        
        print(f"Missing values found in {len(missing_cols)} columns:")
        for col, count in missing_cols.items():
            print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
        
        df_processed = df.copy()
        
        for col in missing_cols.index:
            if strategy == 'auto':
                # Auto strategy based on data type and missing percentage
                missing_pct = missing_cols[col] / len(df)
                
                if df[col].dtype in ['object', 'category']:
                    # Categorical: use mode
                    df_processed[col] = df_processed[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
                elif missing_pct < 0.1:
                    # Low missing: use median for numeric
                    df_processed[col] = df_processed[col].fillna(df[col].median())
                else:
                    # High missing: use KNN imputation
                    if col not in self.imputers:
                        self.imputers[col] = KNNImputer(n_neighbors=5)
                    df_processed[col] = self.imputers[col].fit_transform(df[[col]]).flatten()
            else:
                # Manual strategy
                if strategy == 'mean':
                    df_processed[col] = df_processed[col].fillna(df[col].mean())
                elif strategy == 'median':
                    df_processed[col] = df_processed[col].fillna(df[col].median())
                elif strategy == 'mode':
                    df_processed[col] = df_processed[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
                elif strategy == 'drop':
                    df_processed = df_processed.dropna(subset=[col])
        
        return df_processed
    
    def detect_outliers(self, df, contamination=0.1):
        """Detect outliers using Isolation Forest"""
        print(f"Detecting outliers with contamination: {contamination}")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_mask = np.zeros(len(df), dtype=bool)
        
        for col in numeric_cols:
            if col not in self.outlier_detectors:
                self.outlier_detectors[col] = IsolationForest(
                    contamination=contamination, 
                    random_state=42
                )
            
            col_outliers = self.outlier_detectors[col].fit_predict(df[[col]])
            outlier_mask |= (col_outliers == -1)
        
        outlier_count = outlier_mask.sum()
        print(f"Detected {outlier_count} outliers ({outlier_count/len(df)*100:.1f}%)")
        
        return outlier_mask
    
    def handle_outliers(self, df, method='cap', contamination=0.1):
        """Handle outliers using various methods"""
        print(f"Handling outliers. Method: {method}")
        
        outlier_mask = self.detect_outliers(df, contamination)
        df_processed = df.copy()
        
        if method == 'remove':
            df_processed = df_processed[~outlier_mask]
            print(f"Removed {outlier_mask.sum()} outliers")
        elif method == 'cap':
            # Cap outliers at 1.5 * IQR
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)
            print("Capped outliers using IQR method")
        
        return df_processed
    
    def encode_categorical(self, df, target_col=None):
        """Encode categorical variables"""
        print("Encoding categorical variables")
        
        df_processed = df.copy()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col == target_col:
                continue
                
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df_processed[col] = self.encoders[col].fit_transform(df[col].astype(str))
            else:
                # Handle unseen categories
                unique_values = set(df[col].astype(str).unique())
                known_values = set(self.encoders[col].classes_)
                new_values = unique_values - known_values
                
                if new_values:
                    # Add new categories
                    all_values = list(known_values) + list(new_values)
                    self.encoders[col] = LabelEncoder()
                    self.encoders[col].fit(all_values)
                
                df_processed[col] = self.encoders[col].transform(df[col].astype(str))
        
        return df_processed
    
    def normalize_features(self, df, method='standard', exclude_cols=None):
        """Normalize features using various methods"""
        print(f"Normalizing features. Method: {method}")
        
        if exclude_cols is None:
            exclude_cols = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        df_processed = df.copy()
        
        # Handle NaN values before normalization
        for col in numeric_cols:
            if df_processed[col].isnull().any():
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        
        for col in numeric_cols:
            if method == 'standard':
                if col not in self.scalers:
                    self.scalers[col] = StandardScaler()
                    df_processed[col] = self.scalers[col].fit_transform(df_processed[[col]]).flatten()
                else:
                    df_processed[col] = self.scalers[col].transform(df_processed[[col]]).flatten()
            elif method == 'minmax':
                if col not in self.scalers:
                    self.scalers[col] = MinMaxScaler()
                    df_processed[col] = self.scalers[col].fit_transform(df_processed[[col]]).flatten()
                else:
                    df_processed[col] = self.scalers[col].transform(df_processed[[col]]).flatten()
        
        return df_processed
    
    def preprocess_dataset(self, dataset_name, target_col=None, 
                          missing_strategy='auto', outlier_method='cap'):
        """Complete preprocessing pipeline for a dataset"""
        print(f"\n=== Preprocessing {dataset_name} dataset ===")
        
        # Load dataset
        df = self.load_dataset(dataset_name)
        if df is None:
            return None
        
        print(f"Original shape: {df.shape}")
        
        # Handle missing values
        df = self.handle_missing_values(df, missing_strategy)
        print(f"After missing value handling: {df.shape}")
        
        # Handle outliers
        df = self.handle_outliers(df, outlier_method)
        print(f"After outlier handling: {df.shape}")
        
        # Encode categorical variables
        df = self.encode_categorical(df, target_col)
        print(f"After encoding: {df.shape}")
        
        # Normalize features
        df = self.normalize_features(df, exclude_cols=[target_col] if target_col else [])
        print(f"After normalization: {df.shape}")
        
        # Final cleanup - ensure no NaN values remain
        df = df.fillna(0)
        print(f"After final cleanup: {df.shape}")
        
        print(f"Final shape: {df.shape}")
        return df
    
    def get_preprocessing_summary(self, original_df, processed_df):
        """Get summary of preprocessing changes"""
        summary = {
            'original_shape': original_df.shape,
            'processed_shape': processed_df.shape,
            'missing_values_removed': original_df.isnull().sum().sum() - processed_df.isnull().sum().sum(),
            'columns_processed': len(processed_df.columns),
            'data_types': processed_df.dtypes.value_counts().to_dict()
        }
        return summary

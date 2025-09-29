"""
Machine Learning Models module for AI-Powered Proactive Patient Risk Advisor
Implements Logistic Regression, XGBoost, and LightGBM with cross-validation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class MLModelTrainer:
    """Machine Learning model training and evaluation"""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.cv_scores = {}
        
    def prepare_data(self, df, target_col, test_size=0.2, random_state=42):
        """Prepare data for training with train-test split"""
        print(f"Preparing data for training. Target: {target_col}")
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Handle target encoding if needed
        if y.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
            self.target_encoder = le
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        print(f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
        
        return X_train, X_test, y_train, y_test, X.columns.tolist()
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test, 
                                 model_name='logistic_regression'):
        """Train Logistic Regression model"""
        print(f"\n=== Training {model_name} ===")
        
        # Scale features for logistic regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc'
        )
        
        # Fit model
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Store model and scaler
        self.models[model_name] = model
        self.scalers[model_name] = scaler
        self.cv_scores[model_name] = cv_scores
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"Test AUC: {auc_score:.4f}")
        
        return {
            'model': model,
            'scaler': scaler,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'cv_scores': cv_scores,
            'auc_score': auc_score
        }
    
    def train_xgboost(self, X_train, y_train, X_test, y_test, 
                     model_name='xgboost'):
        """Train XGBoost model"""
        print(f"\n=== Training {model_name} ===")
        
        # XGBoost parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        }
        
        # Train model
        model = xgb.XGBClassifier(**params)
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc'
        )
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Store model
        self.models[model_name] = model
        self.cv_scores[model_name] = cv_scores
        
        # Feature importance
        self.feature_importance[model_name] = dict(zip(
            X_train.columns, model.feature_importances_
        ))
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"Test AUC: {auc_score:.4f}")
        
        return {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'cv_scores': cv_scores,
            'auc_score': auc_score,
            'feature_importance': self.feature_importance[model_name]
        }
    
    def train_lightgbm(self, X_train, y_train, X_test, y_test, 
                      model_name='lightgbm'):
        """Train LightGBM model"""
        print(f"\n=== Training {model_name} ===")
        
        # LightGBM parameters
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'class_weight': 'balanced'
        }
        
        # Train model
        model = lgb.LGBMClassifier(**params)
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc'
        )
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Store model
        self.models[model_name] = model
        self.cv_scores[model_name] = cv_scores
        
        # Feature importance
        self.feature_importance[model_name] = dict(zip(
            X_train.columns, model.feature_importances_
        ))
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"Test AUC: {auc_score:.4f}")
        
        return {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'cv_scores': cv_scores,
            'auc_score': auc_score,
            'feature_importance': self.feature_importance[model_name]
        }
    
    def train_all_models(self, df, target_col, condition_type):
        """Train all models for a specific condition"""
        print(f"\n=== Training all models for {condition_type} ===")
        
        # Prepare data
        X_train, X_test, y_train, y_test, feature_names = self.prepare_data(
            df, target_col, 
            test_size=self.config['MODEL_CONFIG']['test_size'],
            random_state=self.config['MODEL_CONFIG']['random_state']
        )
        
        results = {}
        
        # Train Logistic Regression
        results['logistic_regression'] = self.train_logistic_regression(
            X_train, y_train, X_test, y_test, f'{condition_type}_logistic'
        )
        
        # Train XGBoost
        results['xgboost'] = self.train_xgboost(
            X_train, y_train, X_test, y_test, f'{condition_type}_xgboost'
        )
        
        # Train LightGBM
        results['lightgbm'] = self.train_lightgbm(
            X_train, y_train, X_test, y_test, f'{condition_type}_lightgbm'
        )
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
        print(f"\nBest model: {best_model_name} (AUC: {results[best_model_name]['auc_score']:.4f})")
        
        return results, X_test, y_test, feature_names
    
    def get_model_performance_summary(self, results, X_test, y_test):
        """Get comprehensive performance summary"""
        summary = {}
        
        for model_name, result in results.items():
            summary[model_name] = {
                'auc_score': result['auc_score'],
                'cv_mean': result['cv_scores'].mean(),
                'cv_std': result['cv_scores'].std(),
                'predictions': result['predictions'],
                'probabilities': result['probabilities']
            }
        
        return summary
    
    def save_models(self, condition_type, results):
        """Save trained models"""
        models_dir = self.config['MODELS_DIR']
        models_dir.mkdir(exist_ok=True)
        
        for model_name, result in results.items():
            model_path = models_dir / f"{condition_type}_{model_name}.joblib"
            joblib.dump(result['model'], model_path)
            print(f"Saved model: {model_path}")
            
            # Save scaler if exists
            if 'scaler' in result:
                scaler_path = models_dir / f"{condition_type}_{model_name}_scaler.joblib"
                joblib.dump(result['scaler'], scaler_path)
                print(f"Saved scaler: {scaler_path}")
    
    def load_models(self, condition_type):
        """Load trained models"""
        models_dir = self.config['MODELS_DIR']
        loaded_models = {}
        
        for model_name in ['logistic_regression', 'xgboost', 'lightgbm']:
            model_path = models_dir / f"{condition_type}_{model_name}.joblib"
            if model_path.exists():
                loaded_models[model_name] = joblib.load(model_path)
                print(f"Loaded model: {model_path}")
        
        return loaded_models
    
    def predict_risk(self, model_name, X, return_probability=True):
        """Make predictions using a trained model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Scale features if scaler exists
        if model_name in self.scalers:
            X_scaled = self.scalers[model_name].transform(X)
        else:
            X_scaled = X
        
        if return_probability:
            return model.predict_proba(X_scaled)[:, 1]
        else:
            return model.predict(X_scaled)

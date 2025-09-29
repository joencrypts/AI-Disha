"""
Explainability module for AI-Powered Proactive Patient Risk Advisor
Implements SHAP and LIME explanations for model interpretability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ModelExplainer:
    """Model explainability using SHAP and LIME"""
    
    def __init__(self, config):
        self.config = config
        self.explainers = {}
        self.feature_names = {}
        
    def setup_shap_explainer(self, model, X_train, model_type='tree'):
        """Setup SHAP explainer based on model type"""
        print(f"Setting up SHAP explainer for {model_type} model")
        
        if model_type == 'tree':
            # For tree-based models (XGBoost, LightGBM)
            explainer = shap.TreeExplainer(model)
        elif model_type == 'linear':
            # For linear models (Logistic Regression)
            explainer = shap.LinearExplainer(model, X_train)
        else:
            # For other models, use KernelExplainer
            explainer = shap.KernelExplainer(model.predict_proba, X_train)
        
        return explainer
    
    def setup_lime_explainer(self, X_train, feature_names, class_names=['No Risk', 'Risk']):
        """Setup LIME explainer"""
        print("Setting up LIME explainer")
        
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification',
            discretize_continuous=True
        )
        
        return explainer
    
    def get_shap_values(self, explainer, X_test, max_samples=100):
        """Get SHAP values for test data"""
        print(f"Computing SHAP values for {min(max_samples, len(X_test))} samples")
        
        # Limit samples for performance
        if len(X_test) > max_samples:
            X_sample = X_test.sample(max_samples, random_state=42)
        else:
            X_sample = X_test
        
        shap_values = explainer.shap_values(X_sample)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Get positive class values
        
        return shap_values, X_sample
    
    def plot_shap_summary(self, shap_values, X_sample, feature_names, condition_type, save_path=None):
        """Plot SHAP summary plot"""
        plt.figure(figsize=(10, 8))
        
        # Create SHAP summary plot
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.title(f'SHAP Summary Plot - {condition_type.title()} Risk Prediction')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP summary plot saved to: {save_path}")
        
        plt.show()
    
    def plot_shap_waterfall(self, explainer, X_sample, feature_names, condition_type, 
                           sample_idx=0, save_path=None):
        """Plot SHAP waterfall plot for a single prediction"""
        plt.figure(figsize=(12, 8))
        
        # Get SHAP values for single sample
        shap_values = explainer.shap_values(X_sample.iloc[[sample_idx]])
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Create waterfall plot
        shap.waterfall_plot(
            explainer.expected_value[1] if hasattr(explainer, 'expected_value') else 0,
            shap_values[0],
            X_sample.iloc[sample_idx],
            feature_names=feature_names,
            show=False
        )
        
        plt.title(f'SHAP Waterfall Plot - {condition_type.title()} Risk Prediction (Sample {sample_idx})')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP waterfall plot saved to: {save_path}")
        
        plt.show()
    
    def get_lime_explanation(self, lime_explainer, model, X_sample, feature_names, 
                           sample_idx=0, num_features=10):
        """Get LIME explanation for a single prediction"""
        print(f"Generating LIME explanation for sample {sample_idx}")
        
        # Get prediction
        prediction = model.predict_proba(X_sample.iloc[[sample_idx]])[0]
        
        # Generate explanation
        explanation = lime_explainer.explain_instance(
            X_sample.iloc[sample_idx].values,
            model.predict_proba,
            num_features=num_features
        )
        
        return explanation, prediction
    
    def plot_lime_explanation(self, explanation, condition_type, sample_idx=0, save_path=None):
        """Plot LIME explanation"""
        plt.figure(figsize=(10, 6))
        
        # Create LIME explanation plot
        explanation.as_pyplot_figure()
        plt.title(f'LIME Explanation - {condition_type.title()} Risk Prediction (Sample {sample_idx})')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"LIME explanation plot saved to: {save_path}")
        
        plt.show()
    
    def get_feature_importance_ranking(self, shap_values, feature_names, top_n=20):
        """Get feature importance ranking from SHAP values"""
        # Calculate mean absolute SHAP values
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_shap_values
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def compare_model_explanations(self, models_results, X_test, feature_names, 
                                 condition_type, sample_idx=0):
        """Compare explanations across different models"""
        print(f"Comparing explanations across models for sample {sample_idx}")
        
        sample_data = X_test.iloc[[sample_idx]]
        
        # Get predictions from all models
        predictions = {}
        for model_name, result in models_results.items():
            if 'model' in result:
                predictions[model_name] = result['model'].predict_proba(sample_data)[0][1]
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Model Comparison - {condition_type.title()} Risk Prediction (Sample {sample_idx})', 
                     fontsize=16)
        
        # Plot 1: Model predictions comparison
        axes[0, 0].bar(predictions.keys(), predictions.values())
        axes[0, 0].set_title('Risk Probability by Model')
        axes[0, 0].set_ylabel('Risk Probability')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Feature importance comparison (if available)
        if 'xgboost' in models_results and 'feature_importance' in models_results['xgboost']:
            xgb_importance = models_results['xgboost']['feature_importance']
            top_features = sorted(xgb_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            features, importances = zip(*top_features)
            
            axes[0, 1].barh(features, importances)
            axes[0, 1].set_title('XGBoost Feature Importance (Top 10)')
            axes[0, 1].set_xlabel('Importance')
        
        # Plot 3: Prediction distribution
        all_predictions = [pred for pred in predictions.values()]
        axes[1, 0].hist(all_predictions, bins=10, alpha=0.7)
        axes[1, 0].set_title('Prediction Distribution')
        axes[1, 0].set_xlabel('Risk Probability')
        axes[1, 0].set_ylabel('Frequency')
        
        # Plot 4: Model performance comparison
        model_names = list(predictions.keys())
        model_scores = [models_results[name]['auc_score'] for name in model_names]
        
        axes[1, 1].bar(model_names, model_scores)
        axes[1, 1].set_title('Model AUC Scores')
        axes[1, 1].set_ylabel('AUC Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return predictions
    
    def generate_explanation_report(self, condition_type, model_name, shap_values, 
                                  X_sample, feature_names, sample_idx=0):
        """Generate comprehensive explanation report"""
        print(f"Generating explanation report for {condition_type} - {model_name}")
        
        # Get feature importance ranking
        importance_df = self.get_feature_importance_ranking(shap_values, feature_names)
        
        # Get sample-specific SHAP values
        sample_shap = shap_values[sample_idx]
        sample_data = X_sample.iloc[sample_idx]
        
        # Create explanation text
        explanation_text = f"""
        EXPLANATION REPORT - {condition_type.upper()} RISK PREDICTION
        
        Model: {model_name}
        Sample Index: {sample_idx}
        
        TOP CONTRIBUTING FACTORS:
        """
        
        for i, (_, row) in enumerate(importance_df.head(5).iterrows()):
            feature = row['feature']
            importance = row['importance']
            value = sample_data[feature] if feature in sample_data.index else 'N/A'
            
            explanation_text += f"""
        {i+1}. {feature}: {value} (Impact: {importance:.4f})"""
        
        # Add risk interpretation
        risk_probability = np.sum(sample_shap) + 0.5  # Approximate from SHAP values
        if risk_probability > 0.7:
            risk_level = "HIGH"
        elif risk_probability > 0.4:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"
        
        explanation_text += f"""
        
        OVERALL RISK ASSESSMENT:
        - Risk Level: {risk_level}
        - Risk Probability: {risk_probability:.3f}
        - Key Risk Factors: {', '.join(importance_df.head(3)['feature'].tolist())}
        """
        
        return explanation_text
    
    def save_explanations(self, condition_type, model_name, explanations, save_dir):
        """Save explanation results"""
        save_dir = save_dir / condition_type
        save_dir.mkdir(exist_ok=True)
        
        # Save SHAP values
        shap_path = save_dir / f"{model_name}_shap_values.npy"
        np.save(shap_path, explanations['shap_values'])
        
        # Save feature importance
        importance_path = save_dir / f"{model_name}_feature_importance.csv"
        explanations['importance_df'].to_csv(importance_path, index=False)
        
        print(f"Explanations saved to: {save_dir}")
        
        return save_dir

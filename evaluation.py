"""
Evaluation module for AI-Powered Proactive Patient Risk Advisor
Comprehensive evaluation using AUC-ROC, PR-AUC, sensitivity, F1, calibration, and decision curve analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, f1_score, precision_score, recall_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Comprehensive model evaluation and analysis"""
    
    def __init__(self, config):
        self.config = config
        self.evaluation_results = {}
        
    def evaluate_model(self, y_true, y_pred, y_pred_proba, model_name, condition_type):
        """Comprehensive model evaluation"""
        print(f"Evaluating {model_name} for {condition_type}")
        
        # Basic metrics
        auc_roc = roc_auc_score(y_true, y_pred_proba)
        auc_pr = average_precision_score(y_true, y_pred_proba)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Additional metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Calibration metrics
        brier_score = brier_score_loss(y_true, y_pred_proba)
        
        # Store results
        results = {
            'model_name': model_name,
            'condition_type': condition_type,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'brier_score': brier_score,
            'confusion_matrix': cm,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        self.evaluation_results[f"{condition_type}_{model_name}"] = results
        
        return results
    
    def evaluate_all_models(self, models_results, X_test, y_test, condition_type):
        """Evaluate all models for a condition"""
        print(f"\n=== Evaluating all models for {condition_type} ===")
        
        all_results = {}
        
        for model_name, result in models_results.items():
            if 'model' in result and 'probabilities' in result:
                model_results = self.evaluate_model(
                    y_test, 
                    result['predictions'], 
                    result['probabilities'],
                    model_name, 
                    condition_type
                )
                all_results[model_name] = model_results
        
        return all_results
    
    def create_roc_curves_plot(self, results_dict, condition_type, save_path=None):
        """Create ROC curves comparison plot"""
        fig = go.Figure()
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray')
        ))
        
        # Add ROC curves for each model
        for model_name, results in results_dict.items():
            fpr, tpr, _ = roc_curve(results['y_true'], results['y_pred_proba'])
            auc_score = results['auc_roc']
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{model_name} (AUC={auc_score:.3f})',
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title=f'ROC Curves - {condition_type.title()}',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"ROC curves plot saved to: {save_path}")
        
        return fig
    
    def create_precision_recall_curves_plot(self, results_dict, condition_type, save_path=None):
        """Create Precision-Recall curves comparison plot"""
        fig = go.Figure()
        
        # Add baseline (random classifier)
        baseline_precision = np.mean([results['y_true'] for results in results_dict.values()])
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[baseline_precision, baseline_precision],
            mode='lines',
            name=f'Random Classifier (AP={baseline_precision:.3f})',
            line=dict(dash='dash', color='gray')
        ))
        
        # Add PR curves for each model
        for model_name, results in results_dict.items():
            precision, recall, _ = precision_recall_curve(results['y_true'], results['y_pred_proba'])
            ap_score = results['auc_pr']
            
            fig.add_trace(go.Scatter(
                x=recall, y=precision,
                mode='lines',
                name=f'{model_name} (AP={ap_score:.3f})',
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title=f'Precision-Recall Curves - {condition_type.title()}',
            xaxis_title='Recall',
            yaxis_title='Precision',
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Precision-Recall curves plot saved to: {save_path}")
        
        return fig
    
    def create_calibration_plot(self, results_dict, condition_type, save_path=None):
        """Create calibration plot"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Calibration Plot', 'Reliability Diagram'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        for model_name, results in results_dict.items():
            # Calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                results['y_true'], results['y_pred_proba'], n_bins=10
            )
            
            fig.add_trace(go.Scatter(
                x=mean_predicted_value, y=fraction_of_positives,
                mode='lines+markers',
                name=f'{model_name}',
                line=dict(width=2)
            ), row=1, col=1)
        
        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(dash='dash', color='gray')
        ), row=1, col=1)
        
        # Reliability diagram
        for model_name, results in results_dict.items():
            fraction_of_positives, mean_predicted_value = calibration_curve(
                results['y_true'], results['y_pred_proba'], n_bins=10
            )
            
            fig.add_trace(go.Bar(
                x=mean_predicted_value, y=fraction_of_positives,
                name=f'{model_name}',
                opacity=0.7
            ), row=1, col=2)
        
        fig.update_layout(
            title=f'Model Calibration - {condition_type.title()}',
            width=1200,
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Calibration plot saved to: {save_path}")
        
        return fig
    
    def create_confusion_matrices_plot(self, results_dict, condition_type, save_path=None):
        """Create confusion matrices comparison plot"""
        n_models = len(results_dict)
        fig = make_subplots(
            rows=1, cols=n_models,
            subplot_titles=list(results_dict.keys()),
            specs=[[{"type": "heatmap"}] * n_models]
        )
        
        for i, (model_name, results) in enumerate(results_dict.items(), 1):
            cm = results['confusion_matrix']
            
            fig.add_trace(go.Heatmap(
                z=cm,
                x=['Predicted 0', 'Predicted 1'],
                y=['Actual 0', 'Actual 1'],
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16},
                colorscale='Blues',
                showscale=False
            ), row=1, col=i)
        
        fig.update_layout(
            title=f'Confusion Matrices - {condition_type.title()}',
            width=300 * n_models,
            height=400
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Confusion matrices plot saved to: {save_path}")
        
        return fig
    
    def create_metrics_comparison_plot(self, results_dict, condition_type, save_path=None):
        """Create metrics comparison plot"""
        metrics = ['auc_roc', 'auc_pr', 'f1_score', 'precision', 'recall', 'sensitivity', 'specificity']
        model_names = list(results_dict.keys())
        
        fig = go.Figure()
        
        for metric in metrics:
            values = [results_dict[model][metric] for model in model_names]
            fig.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                x=model_names,
                y=values
            ))
        
        fig.update_layout(
            title=f'Metrics Comparison - {condition_type.title()}',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            width=1000,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Metrics comparison plot saved to: {save_path}")
        
        return fig
    
    def create_decision_curve_analysis(self, results_dict, condition_type, save_path=None):
        """Create decision curve analysis plot"""
        fig = go.Figure()
        
        # Threshold probabilities
        thresholds = np.linspace(0, 1, 101)
        
        for model_name, results in results_dict.items():
            y_true = results['y_true']
            y_pred_proba = results['y_pred_proba']
            
            # Calculate net benefit for different thresholds
            net_benefits = []
            for threshold in thresholds:
                # Predict positive if probability > threshold
                y_pred_thresh = (y_pred_proba > threshold).astype(int)
                
                # Calculate net benefit
                tp = np.sum((y_pred_thresh == 1) & (y_true == 1))
                fp = np.sum((y_pred_thresh == 1) & (y_true == 0))
                fn = np.sum((y_pred_thresh == 0) & (y_true == 1))
                
                if tp + fp > 0:
                    net_benefit = (tp - fp * threshold / (1 - threshold)) / len(y_true)
                else:
                    net_benefit = 0
                
                net_benefits.append(max(0, net_benefit))
            
            fig.add_trace(go.Scatter(
                x=thresholds, y=net_benefits,
                mode='lines',
                name=f'{model_name}',
                line=dict(width=2)
            ))
        
        # Treat all as positive
        treat_all_benefits = []
        for threshold in thresholds:
            tp = np.sum(y_true == 1)
            fp = np.sum(y_true == 0)
            net_benefit = (tp - fp * threshold / (1 - threshold)) / len(y_true)
            treat_all_benefits.append(max(0, net_benefit))
        
        fig.add_trace(go.Scatter(
            x=thresholds, y=treat_all_benefits,
            mode='lines',
            name='Treat All',
            line=dict(dash='dash', color='gray')
        ))
        
        # Treat none
        fig.add_trace(go.Scatter(
            x=thresholds, y=[0] * len(thresholds),
            mode='lines',
            name='Treat None',
            line=dict(dash='dash', color='black')
        ))
        
        fig.update_layout(
            title=f'Decision Curve Analysis - {condition_type.title()}',
            xaxis_title='Threshold Probability',
            yaxis_title='Net Benefit',
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Decision curve analysis saved to: {save_path}")
        
        return fig
    
    def generate_evaluation_report(self, results_dict, condition_type):
        """Generate comprehensive evaluation report"""
        report = f"""
        COMPREHENSIVE MODEL EVALUATION REPORT
        =====================================
        
        Condition Type: {condition_type.upper()}
        Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        """
        
        # Summary table
        report += "\nMODEL PERFORMANCE SUMMARY:\n"
        report += "-" * 80 + "\n"
        report += f"{'Model':<20} {'AUC-ROC':<10} {'AUC-PR':<10} {'F1':<10} {'Precision':<10} {'Recall':<10}\n"
        report += "-" * 80 + "\n"
        
        for model_name, results in results_dict.items():
            report += f"{model_name:<20} {results['auc_roc']:<10.3f} {results['auc_pr']:<10.3f} "
            report += f"{results['f1_score']:<10.3f} {results['precision']:<10.3f} {results['recall']:<10.3f}\n"
        
        # Best model
        best_model = max(results_dict.keys(), key=lambda x: results_dict[x]['auc_roc'])
        report += f"\nBEST MODEL: {best_model} (AUC-ROC: {results_dict[best_model]['auc_roc']:.3f})\n"
        
        # Detailed analysis
        report += "\nDETAILED ANALYSIS:\n"
        report += "-" * 40 + "\n"
        
        for model_name, results in results_dict.items():
            report += f"\n{model_name.upper()}:\n"
            report += f"  AUC-ROC: {results['auc_roc']:.3f}\n"
            report += f"  AUC-PR: {results['auc_pr']:.3f}\n"
            report += f"  F1 Score: {results['f1_score']:.3f}\n"
            report += f"  Precision: {results['precision']:.3f}\n"
            report += f"  Recall: {results['recall']:.3f}\n"
            report += f"  Sensitivity: {results['sensitivity']:.3f}\n"
            report += f"  Specificity: {results['specificity']:.3f}\n"
            report += f"  Brier Score: {results['brier_score']:.3f}\n"
        
        # Recommendations
        report += "\nRECOMMENDATIONS:\n"
        report += "-" * 20 + "\n"
        
        if results_dict[best_model]['auc_roc'] > 0.8:
            report += "✓ Model performance is excellent (AUC-ROC > 0.8)\n"
        elif results_dict[best_model]['auc_roc'] > 0.7:
            report += "✓ Model performance is good (AUC-ROC > 0.7)\n"
        else:
            report += "⚠ Model performance needs improvement (AUC-ROC < 0.7)\n"
        
        if results_dict[best_model]['brier_score'] < 0.25:
            report += "✓ Model calibration is good (Brier Score < 0.25)\n"
        else:
            report += "⚠ Model calibration needs improvement (Brier Score > 0.25)\n"
        
        return report
    
    def save_evaluation_results(self, results_dict, condition_type, save_dir):
        """Save evaluation results to files"""
        save_dir = save_dir / condition_type
        save_dir.mkdir(exist_ok=True)
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            model: {
                'AUC-ROC': results['auc_roc'],
                'AUC-PR': results['auc_pr'],
                'F1-Score': results['f1_score'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'Sensitivity': results['sensitivity'],
                'Specificity': results['specificity'],
                'Brier-Score': results['brier_score']
            }
            for model, results in results_dict.items()
        }).T
        
        metrics_df.to_csv(save_dir / 'evaluation_metrics.csv')
        
        # Save detailed report
        report = self.generate_evaluation_report(results_dict, condition_type)
        with open(save_dir / 'evaluation_report.txt', 'w') as f:
            f.write(report)
        
        print(f"Evaluation results saved to: {save_dir}")
        
        return save_dir

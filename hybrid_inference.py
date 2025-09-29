"""
Hybrid Inference System for AI-Powered Proactive Patient Risk Advisor
Combines ML predictions with knowledge-based system rules
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class HybridInferenceSystem:
    """Hybrid system combining ML predictions with clinical rules"""
    
    def __init__(self, config, ml_models, knowledge_base):
        self.config = config
        self.ml_models = ml_models
        self.knowledge_base = knowledge_base
        self.risk_levels = config['RISK_LEVELS']
        
    def predict_risk_hybrid(self, patient_data: Dict[str, Any], 
                          condition_type: str, 
                          ml_weights: Dict[str, float] = None) -> Dict[str, Any]:
        """Make hybrid risk prediction combining ML and KBS"""
        
        if ml_weights is None:
            ml_weights = {'logistic_regression': 0.3, 'xgboost': 0.4, 'lightgbm': 0.3}
        
        # Get ML predictions
        ml_predictions = self._get_ml_predictions(patient_data, condition_type, ml_weights)
        
        # Get KBS rules evaluation
        kbs_results = self._evaluate_kbs_rules(patient_data, condition_type)
        
        # Combine predictions
        hybrid_result = self._combine_predictions(ml_predictions, kbs_results, condition_type)
        
        return hybrid_result
    
    def _apply_feature_engineering(self, patient_data: Dict[str, Any], condition_type: str) -> Dict[str, Any]:
        """Apply feature engineering to match training data"""
        if condition_type == 'diabetes':
            # Handle gender encoding (convert 0/1 to Female/Male, then to 0/1 as in training)
            if 'gender' in patient_data:
                patient_data['gender'] = {0: 'Female', 1: 'Male'}.get(patient_data['gender'], 'Female')
                patient_data['gender'] = {'Male': 1, 'Female': 0}.get(patient_data['gender'], 0)
            
            # Handle smoking history encoding
            if 'smoking_history' in patient_data:
                smoking_mapping = {
                    'never': 0,
                    'No Info': 1,
                    'former': 2,
                    'current': 3,
                    'not current': 2,
                    'ever': 2
                }
                patient_data['smoking_history'] = smoking_mapping.get(patient_data['smoking_history'], 1)
            
            # Add engineered features
            patient_data['Cardiovascular_Risk_Score'] = 0
            patient_data['Diabetes_Flag'] = 0
            patient_data['Diabetes_Risk_Score'] = 0
            # Note: Don't add 'diabetes' as it's the target variable
            
        elif condition_type == 'kidney':
            # Add missing features for kidney disease
            patient_data['id'] = 1
            patient_data['sg'] = 1.02
            patient_data['al'] = 0
            patient_data['su'] = 0
            patient_data['rbc'] = 0
            patient_data['pc'] = 0
            patient_data['pcc'] = 0
            patient_data['ba'] = 0
            patient_data['cad'] = 0
            patient_data['appet'] = 0
            patient_data['pe'] = 0
            patient_data['ane'] = 0
            patient_data['Cardiovascular_Risk_Score'] = 0
            patient_data['Kidney_Risk_Score'] = 0
            
        elif condition_type == 'heart':
            # Add missing features for heart disease
            patient_data['Heart Disease'] = 0
            
        elif condition_type == 'liver':
            # Add missing features for liver disease
            patient_data['Result'] = 0
        
        return patient_data
    
    def _get_ml_predictions(self, patient_data: Dict[str, Any], 
                           condition_type: str, 
                           ml_weights: Dict[str, float]) -> Dict[str, Any]:
        """Get predictions from all ML models"""
        ml_predictions = {}
        
        # Apply feature engineering to match training data
        patient_data_engineered = self._apply_feature_engineering(patient_data, condition_type)
        
        # Convert patient data to DataFrame for ML models
        patient_df = pd.DataFrame([patient_data_engineered])
        
        for model_name, weight in ml_weights.items():
            model_key = f"{condition_type}_{model_name}"
            
            if model_key in self.ml_models:
                try:
                    model = self.ml_models[model_key]
                    
                    # Get prediction probability
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(patient_df)[0][1]
                    else:
                        prob = model.predict(patient_df)[0]
                    
                    ml_predictions[model_name] = {
                        'probability': prob,
                        'weight': weight,
                        'weighted_probability': prob * weight
                    }
                except Exception as e:
                    print(f"Error getting prediction from {model_name}: {e}")
                    ml_predictions[model_name] = {
                        'probability': 0.5,  # Default neutral probability
                        'weight': weight,
                        'weighted_probability': 0.5 * weight
                    }
        
        # Calculate ensemble prediction
        if ml_predictions:
            ensemble_prob = sum(pred['weighted_probability'] for pred in ml_predictions.values())
            ensemble_prob = min(max(ensemble_prob, 0.0), 1.0)  # Clamp to [0, 1]
        else:
            ensemble_prob = 0.5
        
        ml_predictions['ensemble'] = {
            'probability': ensemble_prob,
            'weight': 1.0,
            'weighted_probability': ensemble_prob
        }
        
        return ml_predictions
    
    def _evaluate_kbs_rules(self, patient_data: Dict[str, Any], 
                           condition_type: str) -> Dict[str, Any]:
        """Evaluate knowledge-based system rules"""
        # Evaluate rules for the specific condition
        matched_rules = self.knowledge_base.evaluate_rules(patient_data, condition_type)
        
        # Also evaluate general health rules
        general_rules = self.knowledge_base.evaluate_rules(patient_data, 'general')
        
        # Combine all matched rules
        all_matched_rules = matched_rules + general_rules
        
        # Determine risk level
        risk_level = self.knowledge_base.get_risk_level(all_matched_rules)
        
        # Generate explanation
        explanation = self.knowledge_base.generate_explanation(all_matched_rules, risk_level)
        
        # Get recommended actions
        recommended_actions = self.knowledge_base.get_recommended_actions(all_matched_rules)
        
        # Get references
        references = self.knowledge_base.get_references(all_matched_rules)
        
        # Convert risk level to probability (for combination with ML)
        risk_level_prob = self._risk_level_to_probability(risk_level)
        
        return {
            'matched_rules': all_matched_rules,
            'risk_level': risk_level,
            'risk_level_probability': risk_level_prob,
            'explanation': explanation,
            'recommended_actions': recommended_actions,
            'references': references
        }
    
    def _risk_level_to_probability(self, risk_level: str) -> float:
        """Convert risk level to probability for combination with ML"""
        risk_level_mapping = {
            'LOW': 0.2,
            'MODERATE': 0.5,
            'HIGH': 0.8
        }
        return risk_level_mapping.get(risk_level, 0.5)
    
    def _combine_predictions(self, ml_predictions: Dict[str, Any], 
                           kbs_results: Dict[str, Any], 
                           condition_type: str) -> Dict[str, Any]:
        """Combine ML predictions with KBS results"""
        
        # Get ensemble ML probability
        ml_prob = ml_predictions.get('ensemble', {}).get('probability', 0.5)
        
        # Get KBS probability
        kbs_prob = kbs_results.get('risk_level_probability', 0.5)
        
        # Weighted combination (can be adjusted based on confidence)
        ml_weight = 0.6  # ML gets 60% weight
        kbs_weight = 0.4  # KBS gets 40% weight
        
        # Combine probabilities
        combined_prob = (ml_prob * ml_weight) + (kbs_prob * kbs_weight)
        
        # Determine final risk level
        final_risk_level = self._probability_to_risk_level(combined_prob)
        
        # Create comprehensive result
        result = {
            'condition_type': condition_type,
            'final_risk_level': final_risk_level,
            'final_risk_probability': combined_prob,
            'ml_predictions': ml_predictions,
            'kbs_results': kbs_results,
            'combination_weights': {
                'ml_weight': ml_weight,
                'kbs_weight': kbs_weight
            },
            'confidence_score': self._calculate_confidence(ml_predictions, kbs_results),
            'explanation': self._generate_hybrid_explanation(ml_predictions, kbs_results, final_risk_level),
            'recommended_actions': kbs_results.get('recommended_actions', []),
            'references': kbs_results.get('references', [])
        }
        
        return result
    
    def _probability_to_risk_level(self, probability: float) -> str:
        """Convert probability to risk level"""
        if probability >= 0.7:
            return 'HIGH'
        elif probability >= 0.4:
            return 'MODERATE'
        else:
            return 'LOW'
    
    def _calculate_confidence(self, ml_predictions: Dict[str, Any], 
                            kbs_results: Dict[str, Any]) -> float:
        """Calculate confidence score for the hybrid prediction"""
        
        # ML confidence based on agreement between models
        ml_probs = [pred['probability'] for pred in ml_predictions.values() 
                   if isinstance(pred, dict) and 'probability' in pred]
        
        if len(ml_probs) > 1:
            ml_std = np.std(ml_probs)
            ml_confidence = max(0, 1 - ml_std)  # Lower std = higher confidence
        else:
            ml_confidence = 0.5
        
        # KBS confidence based on number of matched rules
        kbs_rules_count = len(kbs_results.get('matched_rules', []))
        kbs_confidence = min(1.0, kbs_rules_count / 5)  # Normalize to 0-1
        
        # Combined confidence
        combined_confidence = (ml_confidence * 0.6) + (kbs_confidence * 0.4)
        
        return min(max(combined_confidence, 0.0), 1.0)
    
    def _generate_hybrid_explanation(self, ml_predictions: Dict[str, Any], 
                                   kbs_results: Dict[str, Any], 
                                   final_risk_level: str) -> str:
        """Generate comprehensive hybrid explanation"""
        
        # Get the actual combined probability (not just ML ensemble)
        ml_prob = ml_predictions.get('ensemble', {}).get('probability', 0.5)
        kbs_prob = kbs_results.get('risk_level_probability', 0.5)
        ml_weight = 0.6
        kbs_weight = 0.4
        combined_prob = (ml_prob * ml_weight) + (kbs_prob * kbs_weight)
        
        explanation = f"AI-POWERED RISK ASSESSMENT\n"
        explanation += f"Final Risk Level: {final_risk_level}\n"
        explanation += f"Combined Risk Probability: {combined_prob:.3f}\n\n"
        
        # ML explanation
        explanation += "MACHINE LEARNING MODELS:\n"
        for model_name, pred in ml_predictions.items():
            if isinstance(pred, dict) and 'probability' in pred and model_name != 'ensemble':
                explanation += f"- {model_name.replace('_', ' ').title()}: {pred['probability']:.3f}\n"
        explanation += f"- Ensemble: {ml_prob:.3f}\n"
        
        # KBS explanation
        matched_rules = kbs_results.get('matched_rules', [])
        explanation += f"\nCLINICAL RULES:\n"
        explanation += f"Matched Rules: {len(matched_rules)}\n"
        explanation += f"Clinical Risk Level: {kbs_results.get('risk_level', 'Unknown')}\n"
        
        # Key risk factors from KBS
        if matched_rules:
            explanation += f"\nKey Risk Factors:\n"
            for i, rule in enumerate(matched_rules[:3], 1):  # Top 3 rules
                explanation += f"{i}. {rule['name']} - {rule['explanation']}\n"
        else:
            explanation += f"\nNo specific clinical risk factors identified.\n"
        
        # Confidence
        confidence = self._calculate_confidence(ml_predictions, kbs_results)
        explanation += f"\nConfidence Score: {confidence:.2f}\n"
        
        return explanation
    
    def batch_predict(self, patient_data_list: List[Dict[str, Any]], 
                     condition_type: str) -> List[Dict[str, Any]]:
        """Make predictions for multiple patients"""
        results = []
        
        for i, patient_data in enumerate(patient_data_list):
            try:
                result = self.predict_risk_hybrid(patient_data, condition_type)
                result['patient_id'] = i
                results.append(result)
            except Exception as e:
                print(f"Error processing patient {i}: {e}")
                results.append({
                    'patient_id': i,
                    'error': str(e),
                    'final_risk_level': 'UNKNOWN'
                })
        
        return results
    
    def get_risk_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of risk levels from batch predictions"""
        risk_levels = [result.get('final_risk_level', 'UNKNOWN') for result in results]
        distribution = {}
        
        for risk_level in risk_levels:
            distribution[risk_level] = distribution.get(risk_level, 0) + 1
        
        return distribution
    
    def generate_summary_report(self, results: List[Dict[str, Any]], 
                               condition_type: str) -> str:
        """Generate summary report for batch predictions"""
        
        total_patients = len(results)
        risk_distribution = self.get_risk_distribution(results)
        
        report = f"HYBRID RISK ASSESSMENT SUMMARY\n"
        report += f"Condition: {condition_type.upper()}\n"
        report += f"Total Patients: {total_patients}\n\n"
        
        report += "RISK DISTRIBUTION:\n"
        for risk_level, count in risk_distribution.items():
            percentage = (count / total_patients) * 100
            report += f"- {risk_level}: {count} patients ({percentage:.1f}%)\n"
        
        # High-risk patients
        high_risk_patients = [r for r in results if r.get('final_risk_level') == 'HIGH']
        if high_risk_patients:
            report += f"\nHIGH-RISK PATIENTS REQUIRING IMMEDIATE ATTENTION: {len(high_risk_patients)}\n"
            for patient in high_risk_patients[:5]:  # Show first 5
                report += f"- Patient {patient.get('patient_id', 'Unknown')}: "
                report += f"Risk Probability {patient.get('final_risk_probability', 0):.3f}\n"
        
        return report

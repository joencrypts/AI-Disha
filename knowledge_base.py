"""
Knowledge-Based System module for AI-Powered Proactive Patient Risk Advisor
Implements clinical guidelines and decision rules in JSON format
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class KnowledgeBaseSystem:
    """Knowledge-based system for clinical decision support"""
    
    def __init__(self, config):
        self.config = config
        self.rules = {}
        self.clinical_thresholds = config['CLINICAL_THRESHOLDS']
        self.risk_levels = config['RISK_LEVELS']
        
    def create_clinical_rules(self):
        """Create comprehensive clinical rules based on medical guidelines"""
        
        # Kidney Disease Rules
        kidney_rules = {
            "condition": "Chronic Kidney Disease",
            "rules": [
                {
                    "rule_id": "CKD_001",
                    "name": "Severe CKD by eGFR",
                    "condition": "eGFR < 30",
                    "severity": "HIGH",
                    "explanation": "eGFR below 30 indicates severe kidney dysfunction requiring immediate attention",
                    "recommended_actions": [
                        "Refer to nephrologist immediately",
                        "Consider dialysis preparation",
                        "Monitor fluid balance closely",
                        "Review all medications for kidney clearance"
                    ],
                    "references": ["KDIGO 2012 Clinical Practice Guidelines"],
                    "priority": 1
                },
                {
                    "rule_id": "CKD_002",
                    "name": "Moderate CKD by eGFR",
                    "condition": "30 <= eGFR < 60",
                    "severity": "MODERATE",
                    "explanation": "eGFR between 30-60 indicates moderate kidney dysfunction",
                    "recommended_actions": [
                        "Regular monitoring every 3-6 months",
                        "Control blood pressure < 130/80",
                        "Manage diabetes if present",
                        "Avoid nephrotoxic medications"
                    ],
                    "references": ["KDIGO 2012 Clinical Practice Guidelines"],
                    "priority": 2
                },
                {
                    "rule_id": "CKD_003",
                    "name": "Proteinuria Risk",
                    "condition": "ACR > 30 OR ProteinInUrine > 0.3",
                    "severity": "MODERATE",
                    "explanation": "Elevated protein in urine indicates kidney damage",
                    "recommended_actions": [
                        "ACE inhibitor or ARB therapy",
                        "Monitor proteinuria regularly",
                        "Control blood pressure",
                        "Consider nephrology consultation"
                    ],
                    "references": ["KDIGO 2012 Clinical Practice Guidelines"],
                    "priority": 2
                },
                {
                    "rule_id": "CKD_004",
                    "name": "Hypertension in CKD",
                    "condition": "SystolicBP > 140 AND eGFR < 90",
                    "severity": "MODERATE",
                    "explanation": "High blood pressure accelerates kidney damage",
                    "recommended_actions": [
                        "Blood pressure target < 130/80",
                        "ACE inhibitor or ARB preferred",
                        "Lifestyle modifications",
                        "Regular monitoring"
                    ],
                    "references": ["KDIGO 2012 Clinical Practice Guidelines"],
                    "priority": 2
                }
            ]
        }
        
        # Diabetes Rules
        diabetes_rules = {
            "condition": "Diabetes Mellitus",
            "rules": [
                {
                    "rule_id": "DM_001",
                    "name": "Poor Glycemic Control",
                    "condition": "HbA1c > 8.0 OR blood_glucose_level > 200",
                    "severity": "HIGH",
                    "explanation": "Poor glycemic control increases risk of complications",
                    "recommended_actions": [
                        "Intensify diabetes management",
                        "Consider insulin therapy",
                        "Refer to endocrinologist",
                        "Patient education on self-management"
                    ],
                    "references": ["ADA 2023 Standards of Care"],
                    "priority": 1
                },
                {
                    "rule_id": "DM_002",
                    "name": "Prediabetes Risk",
                    "condition": "5.7 <= HbA1c < 6.5 OR 100 <= blood_glucose_level < 126",
                    "severity": "MODERATE",
                    "explanation": "Prediabetes indicates high risk for developing diabetes",
                    "recommended_actions": [
                        "Lifestyle intervention program",
                        "Weight loss if overweight",
                        "Regular physical activity",
                        "Annual diabetes screening"
                    ],
                    "references": ["ADA 2023 Standards of Care"],
                    "priority": 2
                },
                {
                    "rule_id": "DM_003",
                    "name": "Diabetes with Obesity",
                    "condition": "BMI > 30 AND diabetes == 1",
                    "severity": "MODERATE",
                    "explanation": "Obesity complicates diabetes management",
                    "recommended_actions": [
                        "Weight management program",
                        "Consider GLP-1 receptor agonists",
                        "Bariatric surgery evaluation if appropriate",
                        "Nutritional counseling"
                    ],
                    "references": ["ADA 2023 Standards of Care"],
                    "priority": 2
                }
            ]
        }
        
        # Heart Disease Rules
        heart_rules = {
            "condition": "Cardiovascular Disease",
            "rules": [
                {
                    "rule_id": "CVD_001",
                    "name": "High Cholesterol Risk",
                    "condition": "Cholesterol > 240 OR LDL > 160",
                    "severity": "HIGH",
                    "explanation": "Very high cholesterol levels require immediate intervention",
                    "recommended_actions": [
                        "High-intensity statin therapy",
                        "Lifestyle modifications",
                        "Cardiology consultation",
                        "Regular lipid monitoring"
                    ],
                    "references": ["AHA/ACC 2018 Cholesterol Guidelines"],
                    "priority": 1
                },
                {
                    "rule_id": "CVD_002",
                    "name": "Moderate Cholesterol Risk",
                    "condition": "200 < Cholesterol <= 240 OR 100 < LDL <= 160",
                    "severity": "MODERATE",
                    "explanation": "Elevated cholesterol increases cardiovascular risk",
                    "recommended_actions": [
                        "Moderate-intensity statin therapy",
                        "Dietary modifications",
                        "Regular exercise",
                        "Annual lipid panel"
                    ],
                    "references": ["AHA/ACC 2018 Cholesterol Guidelines"],
                    "priority": 2
                },
                {
                    "rule_id": "CVD_003",
                    "name": "Hypertension Risk",
                    "condition": "BP > 140/90",
                    "severity": "MODERATE",
                    "explanation": "High blood pressure is a major cardiovascular risk factor",
                    "recommended_actions": [
                        "Antihypertensive medication",
                        "DASH diet",
                        "Sodium restriction",
                        "Regular blood pressure monitoring"
                    ],
                    "references": ["AHA/ACC 2017 Hypertension Guidelines"],
                    "priority": 2
                },
                {
                    "rule_id": "CVD_004",
                    "name": "Multiple Risk Factors",
                    "condition": "Age > 55 AND (Cholesterol > 200 OR BP > 140) AND (smoking == 1 OR diabetes == 1)",
                    "severity": "HIGH",
                    "explanation": "Multiple risk factors significantly increase cardiovascular risk",
                    "recommended_actions": [
                        "Comprehensive risk assessment",
                        "Aggressive risk factor modification",
                        "Cardiology consultation",
                        "Regular follow-up"
                    ],
                    "references": ["AHA/ACC 2019 Primary Prevention Guidelines"],
                    "priority": 1
                }
            ]
        }
        
        # Liver Disease Rules
        liver_rules = {
            "condition": "Liver Disease",
            "rules": [
                {
                    "rule_id": "LD_001",
                    "name": "Severe Liver Dysfunction",
                    "condition": "Total Bilirubin > 3.0 OR ALB Albumin < 2.5",
                    "severity": "HIGH",
                    "explanation": "Severe liver dysfunction requires immediate medical attention",
                    "recommended_actions": [
                        "Immediate hepatology consultation",
                        "Liver function monitoring",
                        "Avoid hepatotoxic medications",
                        "Consider liver transplant evaluation"
                    ],
                    "references": ["AASLD 2021 Practice Guidelines"],
                    "priority": 1
                },
                {
                    "rule_id": "LD_002",
                    "name": "Moderate Liver Dysfunction",
                    "condition": "1.2 < Total Bilirubin <= 3.0 OR 2.5 <= ALB Albumin < 3.5",
                    "severity": "MODERATE",
                    "explanation": "Moderate liver dysfunction requires monitoring and intervention",
                    "recommended_actions": [
                        "Regular liver function tests",
                        "Avoid alcohol",
                        "Hepatology consultation",
                        "Monitor for complications"
                    ],
                    "references": ["AASLD 2021 Practice Guidelines"],
                    "priority": 2
                },
                {
                    "rule_id": "LD_003",
                    "name": "Elevated Liver Enzymes",
                    "condition": "Sgpt Alamine Aminotransferase > 56 OR Sgot Aspartate Aminotransferase > 40",
                    "severity": "MODERATE",
                    "explanation": "Elevated liver enzymes indicate liver stress or damage",
                    "recommended_actions": [
                        "Repeat liver function tests",
                        "Investigate underlying cause",
                        "Avoid hepatotoxic substances",
                        "Regular monitoring"
                    ],
                    "references": ["AASLD 2021 Practice Guidelines"],
                    "priority": 2
                },
                {
                    "rule_id": "LD_004",
                    "name": "AST/ALT Ratio Abnormal",
                    "condition": "AST_ALT_Ratio > 2.0",
                    "severity": "MODERATE",
                    "explanation": "High AST/ALT ratio may indicate advanced liver disease",
                    "recommended_actions": [
                        "Further liver evaluation",
                        "Consider imaging studies",
                        "Hepatology consultation",
                        "Monitor for cirrhosis"
                    ],
                    "references": ["AASLD 2021 Practice Guidelines"],
                    "priority": 2
                }
            ]
        }
        
        # General Health Rules
        general_rules = {
            "condition": "General Health",
            "rules": [
                {
                    "rule_id": "GH_001",
                    "name": "Obesity Risk",
                    "condition": "BMI > 30",
                    "severity": "MODERATE",
                    "explanation": "Obesity increases risk for multiple chronic conditions",
                    "recommended_actions": [
                        "Weight management program",
                        "Nutritional counseling",
                        "Regular physical activity",
                        "Regular health monitoring"
                    ],
                    "references": ["WHO Obesity Guidelines"],
                    "priority": 2
                },
                {
                    "rule_id": "GH_002",
                    "name": "Smoking Risk",
                    "condition": "smoking == 1 OR smoking_history == 'current'",
                    "severity": "HIGH",
                    "explanation": "Smoking is a major risk factor for multiple diseases",
                    "recommended_actions": [
                        "Smoking cessation program",
                        "Nicotine replacement therapy",
                        "Behavioral counseling",
                        "Regular follow-up"
                    ],
                    "references": ["USPSTF Smoking Cessation Guidelines"],
                    "priority": 1
                },
                {
                    "rule_id": "GH_003",
                    "name": "Age-Related Risk",
                    "condition": "Age > 65",
                    "severity": "MODERATE",
                    "explanation": "Advanced age increases risk for multiple conditions",
                    "recommended_actions": [
                        "Comprehensive health assessment",
                        "Regular preventive care",
                        "Medication review",
                        "Fall risk assessment"
                    ],
                    "references": ["USPSTF Preventive Care Guidelines"],
                    "priority": 2
                }
            ]
        }
        
        self.rules = {
            "kidney": kidney_rules,
            "diabetes": diabetes_rules,
            "heart": heart_rules,
            "liver": liver_rules,
            "general": general_rules
        }
        
        return self.rules
    
    def save_rules_to_json(self, file_path):
        """Save rules to JSON file"""
        with open(file_path, 'w') as f:
            json.dump(self.rules, f, indent=2)
        print(f"Rules saved to: {file_path}")
    
    def load_rules_from_json(self, file_path):
        """Load rules from JSON file"""
        with open(file_path, 'r') as f:
            self.rules = json.load(f)
        print(f"Rules loaded from: {file_path}")
    
    def evaluate_rules(self, patient_data: Dict[str, Any], condition_type: str) -> List[Dict]:
        """Evaluate rules against patient data"""
        if condition_type not in self.rules:
            return []
        
        matched_rules = []
        condition_rules = self.rules[condition_type]["rules"]
        
        for rule in condition_rules:
            if self._evaluate_condition(rule["condition"], patient_data):
                matched_rules.append(rule)
        
        # Sort by priority (lower number = higher priority)
        matched_rules.sort(key=lambda x: x["priority"])
        
        return matched_rules
    
    def _evaluate_condition(self, condition: str, patient_data: Dict[str, Any]) -> bool:
        """Evaluate a single condition against patient data"""
        try:
            # Replace variable names with actual values
            condition_eval = condition
            for key, value in patient_data.items():
                if isinstance(value, (int, float)):
                    condition_eval = condition_eval.replace(key, str(value))
                elif isinstance(value, str):
                    condition_eval = condition_eval.replace(key, f"'{value}'")
            
            # Evaluate the condition
            return eval(condition_eval)
        except:
            return False
    
    def get_risk_level(self, matched_rules: List[Dict]) -> str:
        """Determine overall risk level based on matched rules"""
        if not matched_rules:
            return "LOW"
        
        # Check for any HIGH severity rules
        high_severity = any(rule["severity"] == "HIGH" for rule in matched_rules)
        moderate_severity = any(rule["severity"] == "MODERATE" for rule in matched_rules)
        
        if high_severity:
            return "HIGH"
        elif moderate_severity:
            return "MODERATE"
        else:
            return "LOW"
    
    def generate_explanation(self, matched_rules: List[Dict], risk_level: str) -> str:
        """Generate human-readable explanation based on matched rules"""
        if not matched_rules:
            return "No specific risk factors identified. Continue regular monitoring."
        
        explanation = f"Risk Level: {risk_level}\n\n"
        explanation += "Identified Risk Factors:\n"
        
        for i, rule in enumerate(matched_rules, 1):
            explanation += f"\n{i}. {rule['name']}\n"
            explanation += f"   - {rule['explanation']}\n"
            explanation += f"   - Severity: {rule['severity']}\n"
        
        explanation += "\nRecommended Actions:\n"
        all_actions = []
        for rule in matched_rules:
            all_actions.extend(rule['recommended_actions'])
        
        # Remove duplicates while preserving order
        unique_actions = list(dict.fromkeys(all_actions))
        for i, action in enumerate(unique_actions, 1):
            explanation += f"{i}. {action}\n"
        
        return explanation
    
    def get_recommended_actions(self, matched_rules: List[Dict]) -> List[str]:
        """Get all recommended actions from matched rules"""
        all_actions = []
        for rule in matched_rules:
            all_actions.extend(rule['recommended_actions'])
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(all_actions))
    
    def get_references(self, matched_rules: List[Dict]) -> List[str]:
        """Get all references from matched rules"""
        all_references = []
        for rule in matched_rules:
            all_references.extend(rule['references'])
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(all_references))
    
    def create_rule_summary(self, condition_type: str) -> Dict:
        """Create summary of rules for a condition"""
        if condition_type not in self.rules:
            return {}
        
        condition_rules = self.rules[condition_type]
        rules = condition_rules["rules"]
        
        summary = {
            "condition": condition_rules["condition"],
            "total_rules": len(rules),
            "high_priority_rules": len([r for r in rules if r["priority"] == 1]),
            "moderate_priority_rules": len([r for r in rules if r["priority"] == 2]),
            "severity_distribution": {
                "HIGH": len([r for r in rules if r["severity"] == "HIGH"]),
                "MODERATE": len([r for r in rules if r["severity"] == "MODERATE"]),
                "LOW": len([r for r in rules if r["severity"] == "LOW"])
            }
        }
        
        return summary

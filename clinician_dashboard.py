"""
Clinician Dashboard module for AI-Powered Proactive Patient Risk Advisor
Interactive dashboard for healthcare providers
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class ClinicianDashboard:
    """Interactive dashboard for healthcare providers"""
    
    def __init__(self, config):
        self.config = config
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.patient_data = {}
        self.risk_assessments = {}
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("AI-Powered Proactive Patient Risk Advisor", 
                           className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Control Panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Control Panel"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Condition Type:"),
                                    dcc.Dropdown(
                                        id='condition-dropdown',
                                        options=[
                                            {'label': 'Kidney Disease', 'value': 'kidney'},
                                            {'label': 'Diabetes', 'value': 'diabetes'},
                                            {'label': 'Heart Disease', 'value': 'heart'},
                                            {'label': 'Liver Disease', 'value': 'liver'}
                                        ],
                                        value='kidney'
                                    )
                                ], width=6),
                                dbc.Col([
                                    html.Label("Patient ID:"),
                                    dcc.Dropdown(
                                        id='patient-dropdown',
                                        options=[],
                                        value=None
                                    )
                                ], width=6)
                            ]),
                            html.Br(),
                            dbc.Button("Refresh Data", id="refresh-btn", 
                                     color="primary", className="me-2"),
                            dbc.Button("Generate Voice Alert", id="voice-alert-btn", 
                                     color="warning", className="me-2"),
                            dbc.Button("Export Report", id="export-btn", 
                                     color="success")
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Risk Overview Cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("High Risk", className="text-danger"),
                            html.H2(id="high-risk-count", className="text-danger")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Moderate Risk", className="text-warning"),
                            html.H2(id="moderate-risk-count", className="text-warning")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Low Risk", className="text-success"),
                            html.H2(id="low-risk-count", className="text-success")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Total Patients", className="text-info"),
                            html.H2(id="total-patients-count", className="text-info")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            # Main Content Tabs
            dbc.Row([
                dbc.Col([
                    dbc.Tabs([
                        dbc.Tab(label="Patient Overview", tab_id="overview"),
                        dbc.Tab(label="Risk Analysis", tab_id="risk-analysis"),
                        dbc.Tab(label="Lab Trends", tab_id="lab-trends"),
                        dbc.Tab(label="Model Performance", tab_id="model-performance"),
                        dbc.Tab(label="Voice Assistant", tab_id="voice-assistant")
                    ], id="main-tabs", active_tab="overview")
                ], width=12)
            ]),
            
            # Tab Content
            dbc.Row([
                dbc.Col([
                    html.Div(id="tab-content")
                ], width=12)
            ], className="mt-4")
            
        ], fluid=True)
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output('patient-dropdown', 'options'),
             Output('patient-dropdown', 'value')],
            [Input('condition-dropdown', 'value'),
             Input('refresh-btn', 'n_clicks')]
        )
        def update_patient_dropdown(condition, refresh_clicks):
            # This would typically load from database
            # For demo purposes, generate sample patient IDs
            patient_options = [
                {'label': f'Patient {i:03d}', 'value': f'patient_{i:03d}'}
                for i in range(1, 21)
            ]
            return patient_options, patient_options[0]['value'] if patient_options else None
        
        @self.app.callback(
            [Output('high-risk-count', 'children'),
             Output('moderate-risk-count', 'children'),
             Output('low-risk-count', 'children'),
             Output('total-patients-count', 'children')],
            [Input('condition-dropdown', 'value'),
             Input('refresh-btn', 'n_clicks')]
        )
        def update_risk_counts(condition, refresh_clicks):
            # Sample data - in real implementation, this would come from database
            risk_counts = {
                'kidney': {'high': 5, 'moderate': 12, 'low': 8, 'total': 25},
                'diabetes': {'high': 3, 'moderate': 15, 'low': 7, 'total': 25},
                'heart': {'high': 7, 'moderate': 10, 'low': 8, 'total': 25},
                'liver': {'high': 4, 'moderate': 11, 'low': 10, 'total': 25}
            }
            
            counts = risk_counts.get(condition, {'high': 0, 'moderate': 0, 'low': 0, 'total': 0})
            return counts['high'], counts['moderate'], counts['low'], counts['total']
        
        @self.app.callback(
            Output('tab-content', 'children'),
            [Input('main-tabs', 'active_tab'),
             Input('condition-dropdown', 'value'),
             Input('patient-dropdown', 'value')]
        )
        def update_tab_content(active_tab, condition, patient_id):
            if active_tab == "overview":
                return self.create_overview_tab(condition, patient_id)
            elif active_tab == "risk-analysis":
                return self.create_risk_analysis_tab(condition, patient_id)
            elif active_tab == "lab-trends":
                return self.create_lab_trends_tab(condition, patient_id)
            elif active_tab == "model-performance":
                return self.create_model_performance_tab(condition)
            elif active_tab == "voice-assistant":
                return self.create_voice_assistant_tab(condition, patient_id)
            else:
                return html.Div("Select a tab to view content")
    
    def create_overview_tab(self, condition, patient_id):
        """Create patient overview tab"""
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Patient Risk Card"),
                    dbc.CardBody([
                        html.H5(f"Patient: {patient_id}"),
                        html.H6(f"Condition: {condition.title()}"),
                        dbc.Alert("HIGH RISK", color="danger", className="mb-3"),
                        html.P("Risk Probability: 0.85"),
                        html.P("Last Updated: 2024-01-15 14:30"),
                        html.Hr(),
                        html.H6("Key Risk Factors:"),
                        html.Ul([
                            html.Li("eGFR < 30 (Severe kidney dysfunction)"),
                            html.Li("Proteinuria > 300 mg/g"),
                            html.Li("Hypertension (BP > 140/90)"),
                            html.Li("Age > 65 years")
                        ]),
                        html.Hr(),
                        html.H6("Recommended Actions:"),
                        html.Ul([
                            html.Li("Immediate nephrology consultation"),
                            html.Li("ACE inhibitor therapy"),
                            html.Li("Blood pressure management"),
                            html.Li("Regular monitoring")
                        ])
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Risk Trend"),
                    dbc.CardBody([
                        dcc.Graph(
                            figure=self.create_risk_trend_chart(patient_id)
                        )
                    ])
                ])
            ], width=6)
        ])
    
    def create_risk_analysis_tab(self, condition, patient_id):
        """Create risk analysis tab"""
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Risk Distribution"),
                    dbc.CardBody([
                        dcc.Graph(
                            figure=self.create_risk_distribution_chart(condition)
                        )
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Feature Importance"),
                    dbc.CardBody([
                        dcc.Graph(
                            figure=self.create_feature_importance_chart(condition)
                        )
                    ])
                ])
            ], width=6)
        ])
    
    def create_lab_trends_tab(self, condition, patient_id):
        """Create lab trends tab"""
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Laboratory Trends"),
                    dbc.CardBody([
                        dcc.Graph(
                            figure=self.create_lab_trends_chart(patient_id, condition)
                        )
                    ])
                ])
            ], width=12)
        ])
    
    def create_model_performance_tab(self, condition):
        """Create model performance tab"""
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Model Performance Metrics"),
                    dbc.CardBody([
                        dcc.Graph(
                            figure=self.create_model_performance_chart(condition)
                        )
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("ROC Curves"),
                    dbc.CardBody([
                        dcc.Graph(
                            figure=self.create_roc_curves_chart(condition)
                        )
                    ])
                ])
            ], width=6)
        ])
    
    def create_voice_assistant_tab(self, condition, patient_id):
        """Create voice assistant tab"""
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Voice Assistant"),
                    dbc.CardBody([
                        html.H5("Patient Interaction"),
                        html.P("Click the buttons below to interact with the voice assistant:"),
                        dbc.ButtonGroup([
                            dbc.Button("Start Consultation", color="primary", className="me-2"),
                            dbc.Button("Generate Response", color="success", className="me-2"),
                            dbc.Button("Play Audio", color="info")
                        ]),
                        html.Hr(),
                        html.H5("Generated Script:"),
                        html.Div(id="voice-script", className="border p-3"),
                        html.Hr(),
                        html.H5("Audio Files:"),
                        html.Ul([
                            html.Li(html.A("Initial Consultation", href="#", className="me-2")),
                            html.Li(html.A("Risk Assessment Response", href="#", className="me-2")),
                            html.Li(html.A("Emergency Alert", href="#", className="me-2"))
                        ])
                    ])
                ])
            ], width=12)
        ])
    
    def create_risk_trend_chart(self, patient_id):
        """Create risk trend chart"""
        # Sample data
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        risk_scores = np.random.uniform(0.3, 0.9, 30)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=risk_scores,
            mode='lines+markers',
            name='Risk Score',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=f"Risk Trend - {patient_id}",
            xaxis_title="Date",
            yaxis_title="Risk Score",
            height=400
        )
        
        return fig
    
    def create_risk_distribution_chart(self, condition):
        """Create risk distribution chart"""
        risk_levels = ['Low', 'Moderate', 'High']
        counts = [8, 12, 5]  # Sample data
        
        fig = go.Figure(data=[
            go.Bar(x=risk_levels, y=counts, 
                  marker_color=['green', 'orange', 'red'])
        ])
        
        fig.update_layout(
            title=f"Risk Distribution - {condition.title()}",
            xaxis_title="Risk Level",
            yaxis_title="Number of Patients",
            height=400
        )
        
        return fig
    
    def create_feature_importance_chart(self, condition):
        """Create feature importance chart"""
        # Sample feature importance data
        features = ['eGFR', 'Age', 'BMI', 'SystolicBP', 'Creatinine', 
                   'Proteinuria', 'Cholesterol', 'Diabetes']
        importance = [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04]
        
        fig = go.Figure(data=[
            go.Bar(x=importance, y=features, orientation='h',
                  marker_color='lightblue')
        ])
        
        fig.update_layout(
            title=f"Feature Importance - {condition.title()}",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=400
        )
        
        return fig
    
    def create_lab_trends_chart(self, patient_id, condition):
        """Create lab trends chart"""
        dates = pd.date_range('2024-01-01', periods=12, freq='M')
        
        if condition == 'kidney':
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('eGFR', 'Creatinine', 'Proteinuria', 'BUN'),
                vertical_spacing=0.1
            )
            
            # eGFR
            fig.add_trace(go.Scatter(x=dates, y=np.random.uniform(20, 80, 12), 
                                   name='eGFR'), row=1, col=1)
            # Creatinine
            fig.add_trace(go.Scatter(x=dates, y=np.random.uniform(0.8, 3.5, 12), 
                                   name='Creatinine'), row=1, col=2)
            # Proteinuria
            fig.add_trace(go.Scatter(x=dates, y=np.random.uniform(0, 500, 12), 
                                   name='Proteinuria'), row=2, col=1)
            # BUN
            fig.add_trace(go.Scatter(x=dates, y=np.random.uniform(10, 50, 12), 
                                   name='BUN'), row=2, col=2)
        
        fig.update_layout(
            title=f"Laboratory Trends - {patient_id}",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_model_performance_chart(self, condition):
        """Create model performance chart"""
        models = ['Logistic Regression', 'XGBoost', 'LightGBM']
        auc_scores = [0.82, 0.89, 0.87]
        
        fig = go.Figure(data=[
            go.Bar(x=models, y=auc_scores, marker_color='lightgreen')
        ])
        
        fig.update_layout(
            title=f"Model Performance - {condition.title()}",
            xaxis_title="Model",
            yaxis_title="AUC Score",
            height=400
        )
        
        return fig
    
    def create_roc_curves_chart(self, condition):
        """Create ROC curves chart"""
        # Sample ROC curve data
        fpr = np.linspace(0, 1, 100)
        tpr_lr = 0.8 * fpr + 0.2
        tpr_xgb = 0.9 * fpr + 0.1
        tpr_lgb = 0.85 * fpr + 0.15
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=fpr, y=tpr_lr, mode='lines', 
                               name='Logistic Regression (AUC=0.82)'))
        fig.add_trace(go.Scatter(x=fpr, y=tpr_xgb, mode='lines', 
                               name='XGBoost (AUC=0.89)'))
        fig.add_trace(go.Scatter(x=fpr, y=tpr_lgb, mode='lines', 
                               name='LightGBM (AUC=0.87)'))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                               name='Random Classifier', 
                               line=dict(dash='dash', color='gray')))
        
        fig.update_layout(
            title=f"ROC Curves - {condition.title()}",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=400
        )
        
        return fig
    
    def run_dashboard(self, debug=True, port=8050):
        """Run the dashboard"""
        print(f"Starting dashboard on port {port}")
        self.app.run_server(debug=debug, port=port)

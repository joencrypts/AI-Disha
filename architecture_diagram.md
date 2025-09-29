# AI-Powered Proactive Patient Risk Advisor - System Architecture

## High-Level Architecture

```mermaid
graph TB
    subgraph "Data Sources"
        A1[Kidney Dataset]
        A2[Diabetes Dataset]
        A3[Heart Dataset]
        A4[Liver Dataset]
    end
    
    subgraph "Data Processing Layer"
        B1[Data Preprocessing]
        B2[Feature Engineering]
        B3[Missing Value Handling]
        B4[Outlier Detection]
    end
    
    subgraph "Machine Learning Layer"
        C1[Logistic Regression]
        C2[XGBoost]
        C3[LightGBM]
        C4[Cross-Validation]
    end
    
    subgraph "Knowledge Base Layer"
        D1[Clinical Rules Engine]
        D2[Risk Stratification]
        D3[Evidence-Based Guidelines]
        D4[Actionable Recommendations]
    end
    
    subgraph "Hybrid Inference Engine"
        E1[ML Predictions]
        E2[KBS Rules]
        E3[Risk Combination]
        E4[Confidence Scoring]
    end
    
    subgraph "Explainability Layer"
        F1[SHAP Explanations]
        F2[LIME Interpretations]
        F3[Feature Importance]
        F4[Model Interpretability]
    end
    
    subgraph "Interface Layer"
        G1[Voice Assistant]
        G2[Clinician Dashboard]
        G3[Patient Interaction]
        G4[Emergency Alerts]
    end
    
    subgraph "Evaluation Layer"
        H1[AUC-ROC Analysis]
        H2[Precision-Recall Curves]
        H3[Calibration Assessment]
        H4[Decision Curve Analysis]
    end
    
    A1 --> B1
    A2 --> B1
    A3 --> B1
    A4 --> B1
    
    B1 --> B2
    B2 --> B3
    B3 --> B4
    
    B4 --> C1
    B4 --> C2
    B4 --> C3
    C1 --> C4
    C2 --> C4
    C3 --> C4
    
    B4 --> D1
    D1 --> D2
    D2 --> D3
    D3 --> D4
    
    C4 --> E1
    D4 --> E2
    E1 --> E3
    E2 --> E3
    E3 --> E4
    
    E1 --> F1
    E1 --> F2
    E1 --> F3
    E1 --> F4
    
    E4 --> G1
    E4 --> G2
    E4 --> G3
    E4 --> G4
    
    C4 --> H1
    C4 --> H2
    C4 --> H3
    C4 --> H4
```

## Data Flow Diagram

```mermaid
flowchart TD
    subgraph "Input"
        I1[Patient Data]
        I2[Clinical Parameters]
        I3[Lab Values]
        I4[Symptoms]
    end
    
    subgraph "Processing"
        P1[Data Validation]
        P2[Feature Extraction]
        P3[Risk Calculation]
        P4[Rule Matching]
    end
    
    subgraph "ML Pipeline"
        M1[Model Training]
        M2[Cross-Validation]
        M3[Prediction]
        M4[Probability Calibration]
    end
    
    subgraph "Knowledge Base"
        K1[Rule Evaluation]
        K2[Risk Stratification]
        K3[Action Generation]
        K4[Reference Lookup]
    end
    
    subgraph "Hybrid System"
        H1[ML Weighting]
        H2[KBS Weighting]
        H3[Risk Combination]
        H4[Confidence Calculation]
    end
    
    subgraph "Output"
        O1[Risk Level]
        O2[Risk Probability]
        O3[Explanations]
        O4[Recommendations]
        O5[Voice Response]
        O6[Dashboard Display]
    end
    
    I1 --> P1
    I2 --> P1
    I3 --> P1
    I4 --> P1
    
    P1 --> P2
    P2 --> P3
    P3 --> P4
    
    P2 --> M1
    M1 --> M2
    M2 --> M3
    M3 --> M4
    
    P4 --> K1
    K1 --> K2
    K2 --> K3
    K3 --> K4
    
    M4 --> H1
    K4 --> H2
    H1 --> H3
    H2 --> H3
    H3 --> H4
    
    H4 --> O1
    H4 --> O2
    H4 --> O3
    H4 --> O4
    H4 --> O5
    H4 --> O6
```

## Component Interaction Diagram

```mermaid
sequenceDiagram
    participant P as Patient
    participant VA as Voice Assistant
    participant H as Hybrid System
    participant ML as ML Models
    participant KB as Knowledge Base
    participant CD as Clinician Dashboard
    participant E as Evaluation System
    
    P->>VA: Reports symptoms
    VA->>H: Request risk assessment
    H->>ML: Get ML predictions
    ML-->>H: Risk probabilities
    H->>KB: Evaluate clinical rules
    KB-->>H: Matched rules & actions
    H->>H: Combine predictions & rules
    H-->>VA: Hybrid risk assessment
    VA->>VA: Generate voice response
    VA-->>P: Audio response & recommendations
    
    H->>CD: Update dashboard data
    CD->>E: Request evaluation metrics
    E-->>CD: Performance statistics
    CD->>CD: Update visualizations
    
    Note over H: High-risk patients trigger alerts
    H->>CD: Emergency alert
    CD->>VA: Generate emergency voice alert
```

## Technology Stack

```mermaid
graph LR
    subgraph "Frontend"
        F1[Dash Dashboard]
        F2[Plotly Visualizations]
        F3[HTML/CSS/JS]
    end
    
    subgraph "Backend"
        B1[Python 3.8+]
        B2[Flask API]
        B3[Pandas/NumPy]
        B4[Scikit-learn]
    end
    
    subgraph "ML Libraries"
        M1[XGBoost]
        M2[LightGBM]
        M3[SHAP]
        M4[LIME]
    end
    
    subgraph "External Services"
        E1[ElevenLabs API]
        E2[Voice Synthesis]
        E3[Audio Processing]
    end
    
    subgraph "Data Storage"
        D1[CSV Files]
        D2[JSON Rules]
        D3[Model Files]
        D4[Results Cache]
    end
    
    F1 --> B1
    F2 --> B1
    F3 --> B1
    
    B1 --> B2
    B2 --> B3
    B3 --> B4
    
    B4 --> M1
    B4 --> M2
    B4 --> M3
    B4 --> M4
    
    B1 --> E1
    E1 --> E2
    E2 --> E3
    
    B1 --> D1
    B1 --> D2
    B1 --> D3
    B1 --> D4
```

## Risk Assessment Workflow

```mermaid
flowchart TD
    Start([Patient Data Input]) --> Validate{Data Validation}
    Validate -->|Valid| Preprocess[Data Preprocessing]
    Validate -->|Invalid| Error[Return Error]
    
    Preprocess --> Features[Feature Engineering]
    Features --> ML[ML Model Prediction]
    Features --> Rules[Knowledge Base Rules]
    
    ML --> MLProb[ML Risk Probability]
    Rules --> RuleRisk[Rule-based Risk Level]
    
    MLProb --> Hybrid[Hybrid Inference]
    RuleRisk --> Hybrid
    
    Hybrid --> RiskLevel[Final Risk Level]
    Hybrid --> RiskProb[Final Risk Probability]
    Hybrid --> Confidence[Confidence Score]
    
    RiskLevel --> Explain[Generate Explanations]
    RiskProb --> Explain
    Confidence --> Explain
    
    Explain --> Voice[Voice Assistant]
    Explain --> Dashboard[Clinician Dashboard]
    Explain --> Alert{Emergency Check}
    
    Alert -->|High Risk| Emergency[Emergency Alert]
    Alert -->|Normal| Normal[Standard Response]
    
    Emergency --> Output[Final Output]
    Normal --> Output
    Voice --> Output
    Dashboard --> Output
    
    Output --> End([End])
```

## Model Training Pipeline

```mermaid
flowchart TD
    Data[Raw Dataset] --> Split[Train/Test Split]
    Split --> CV[Cross-Validation]
    
    CV --> LR[Logistic Regression]
    CV --> XGB[XGBoost]
    CV --> LGB[LightGBM]
    
    LR --> Eval1[Model Evaluation]
    XGB --> Eval2[Model Evaluation]
    LGB --> Eval3[Model Evaluation]
    
    Eval1 --> Metrics1[Performance Metrics]
    Eval2 --> Metrics2[Performance Metrics]
    Eval3 --> Metrics3[Performance Metrics]
    
    Metrics1 --> Compare[Model Comparison]
    Metrics2 --> Compare
    Metrics3 --> Compare
    
    Compare --> Best[Select Best Model]
    Best --> Save[Save Model]
    Save --> Deploy[Deploy for Inference]
```

This architecture provides a comprehensive, scalable, and maintainable system for AI-powered healthcare risk assessment with multiple layers of validation, explanation, and user interaction.

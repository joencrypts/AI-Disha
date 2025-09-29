"""
Script to run the AI-Powered Proactive Patient Risk Advisor Dashboard
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def run_dashboard():
    """Run the clinician dashboard"""
    try:
        from main import AIHealthcareSystem
        
        print("="*60)
        print("AI-POWERED PROACTIVE PATIENT RISK ADVISOR")
        print("CLINICIAN DASHBOARD")
        print("="*60)
        
        # Initialize system
        print("Initializing AI Healthcare System...")
        system = AIHealthcareSystem()
        
        # Setup knowledge base
        print("Setting up Knowledge Base...")
        system.setup_knowledge_base()
        
        print("\n" + "="*60)
        print("STARTING CLINICIAN DASHBOARD")
        print("="*60)
        print("Dashboard will be available at: http://localhost:8050")
        print("Press Ctrl+C to stop the dashboard")
        print("="*60)
        
        # Run dashboard
        system.run_dashboard(debug=True, port=8050)
        
    except KeyboardInterrupt:
        print("\nDashboard stopped by user.")
    except Exception as e:
        print(f"Error running dashboard: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_dashboard()

#!/usr/bin/env python3
"""
Startup script for AI-Powered Proactive Patient Risk Advisor
with Meditex Template Integration
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'flask',
        'pandas',
        'numpy',
        'scikit-learn',
        'joblib',
        'matplotlib',
        'seaborn',
        'plotly'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Installing missing packages...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✅ All packages installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please install manually:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
    else:
        print("✅ All required packages are installed!")
    
    return True

def check_directories():
    """Check if required directories exist"""
    required_dirs = [
        'static/assets/css',
        'static/assets/js', 
        'static/assets/image',
        'static/assets/fonts',
        'templates',
        'data',
        'models',
        'results'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print("❌ Missing required directories:")
        for dir_path in missing_dirs:
            print(f"   - {dir_path}")
        print("\n📁 Creating missing directories...")
        for dir_path in missing_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"   ✅ Created: {dir_path}")
    else:
        print("✅ All required directories exist!")
    
    return True

def check_files():
    """Check if required files exist"""
    required_files = [
        'app.py',
        'main.py',
        'config.py',
        'templates/base.html',
        'templates/index.html',
        'templates/risk_assessment.html'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    else:
        print("✅ All required files exist!")
        return True

def start_application():
    """Start the Flask application"""
    print("\n🚀 Starting AI Healthcare Risk Advisor...")
    print("="*60)
    print("🌐 Web Application: http://localhost:5000")
    print("📊 Clinician Dashboard: http://localhost:8050")
    print("🔧 Press Ctrl+C to stop the server")
    print("="*60)
    
    try:
        # Import and run the Flask app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except ImportError as e:
        print(f"❌ Error importing app: {e}")
        print("Please make sure app.py exists and is properly configured.")
        return False
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        return False

def main():
    """Main startup function"""
    print("🏥 AI-Powered Proactive Patient Risk Advisor")
    print("🎨 With Meditex Template Integration")
    print("="*60)
    
    # Check system requirements
    print("\n📋 Checking system requirements...")
    
    if not check_requirements():
        print("\n❌ System requirements check failed!")
        return False
    
    if not check_directories():
        print("\n❌ Directory check failed!")
        return False
    
    if not check_files():
        print("\n❌ File check failed!")
        print("Please ensure all required files are present.")
        return False
    
    print("\n✅ All checks passed! Starting application...")
    
    # Start the application
    start_application()
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user.")
        print("Thank you for using AI Healthcare Risk Advisor!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check the error and try again.")

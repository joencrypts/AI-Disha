from setuptools import setup, find_packages

setup(
    name="wellora-ai-healthcare",
    version="1.0.0",
    description="AI-Powered Healthcare Risk Assessment Platform",
    author="Wellora Team",
    packages=find_packages(),
    install_requires=[
        "Flask",
        "Werkzeug", 
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "joblib",
        "google-generativeai",
        "requests",
        "python-dotenv",
        "gunicorn"
    ],
    python_requires=">=3.11",
)

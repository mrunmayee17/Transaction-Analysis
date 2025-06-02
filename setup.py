from setuptools import setup, find_packages

setup(
    name="credit_fraud_detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "requests",
        "python-arango",
        "networkx",
        "email-validator",
        "pandas",
        "numpy",
        "scikit-learn",
        "python-dotenv",
        "pydantic",
    ],
) 
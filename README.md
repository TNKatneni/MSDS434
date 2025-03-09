# Housing Prediction Project 🏠

## Final Documentation Reflection Paper 
Google Drive Link: https://drive.google.com/file/d/1eAvfH_inHBoWLcpYsWVdqo4GPEsJd9h6/view?usp=sharing

## Video Demonstrations
Google Drive Link: https://drive.google.com/drive/folders/1ViCtnuH5xBPZPYHfUDzNVYKBDCdlXLeP?usp=drive_link

## Overview
- This project aims to analyze housing market data to develop predictive models for home price trends. It includes data preprocessing, model training, and deployment using cloud-based services.

- This project is deployed on AWS Elastic Beanstalk, with the predictive model hosted on AWS SageMaker. The API is publicly accessible (for a limited time), allowing users to test housing price predictions without needing to deploy the application themselves.

## Project Structure
- **.github/workflows/**
- test.yml – CI/CD workflow for testing (if applicable)
- **app/**
- .ebextensions/ – Elastic Beanstalk configuration
- prometheus.config – Prometheus monitoring config
- application.py – Flask API application
- requirements.txt – Required dependencies
- **data/**
- raw_housing_data.csv – Original dataset
- sampled_data.csv – Processed dataset for modeling
- **scripts/**
- app_bundle.zip – Deployment bundle for AWS Elastic Beanstalk
- deploy.py – Deployment automation script
- destroy.py – Cleanup script for AWS resources
- preprocess.py – Data preprocessing script
- .gitignore – Ignore unnecessary files
- README.md – Project documentation
- test.ipynb – Jupyter Notebook for API testing


## API Testing Instructions 
This Jupyter Notebook (`test.ipynb`) provides an interactive way to test the housing prediction API. It includes functions to check whether the API is running and send test data to get a predicted house price. To use the notebook, install the required dependencies with `pip install requests notebook`, then open it using `jupyter notebook test.ipynb`. Running the cells will verify the API status and send a sample request to the prediction endpoint. Below is the full notebook content for reference.
```python
import requests
import json

API_URL = "http://housing-env-tarun.eba-erjum2zi.us-east-1.elasticbeanstalk.com/"

# Check API status
def check_api_status():
    response = requests.get(f"{API_URL}/")
    if response.status_code == 200:
        print("✅ API is running:", response.json())
    else:
        print("❌ API might not be running:", response.status_code, response.text)

# Test prediction request (Alter the payload values for testing)
def test_prediction():
    payload = {
        "bedrooms": 1,
        "bathrooms": 2,
        "lot_size": 0.2,
        "house_size": 1500
    }
    
    response = requests.post(f"{API_URL}/predict", json=payload)
    
    if response.status_code == 200:
        print("✅ Prediction successful:", response.json())
    else:
        print("❌ Prediction failed:", response.status_code, response.text)

print("Checking API Status...")
check_api_status()

print("\nTesting Prediction Endpoint...")
test_prediction()

```


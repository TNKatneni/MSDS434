# Housing Prediction Project 🏠

## Final Documentation Reflection Paper (1-page)
Google Drive Link: [PDF](https://drive.google.com/file/d/1eAvfH_inHBoWLcpYsWVdqo4GPEsJd9h6/view?usp=sharing)

## Final Project Reflection (2-page)
Google Drive Link: [PDF](https://drive.google.com/file/d/1KAadpWYIohdyWb7d9VmmMN-TcwMzEYJS/view?usp=sharing) 

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
- 01_environment.config - AWS config
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
This Jupyter Notebook (`test.ipynb`) provides an interactive way to test the housing prediction API. It includes functions to check whether the API is running and send test data to get a predicted house price. To use the notebook, simply copy and paste the code snippets into a jupyter notebook file and run. Running the first snippet will verify the API status. Running the second snippet will send a  request to the prediction endpoint. Below is the full notebook content for reference.

### Health Check 
```python
import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_URL = os.environ.get("API_URL")
if not API_URL:
    raise ValueError("API_URL is not set in the environment.")

# Check API status
def check_api_status():
    response = requests.get(f"{API_URL}/")
    if response.status_code == 200:
        print("✅ API is running:", response.json())
    else:
        print("❌ API might not be running:", response.status_code, response.text)


print("Checking API Status...")
check_api_status()
```

### Prediction Request 
```python
# Test prediction request (Alter the payload values for testing)
def test_prediction():
    payload = {
        "bedrooms": 6,
        "bathrooms": 4,
        "lot_size": 1.0,
        "house_size": 3000
    }
    
    response = requests.post(f"{API_URL}/predict", json=payload)
    
    if response.status_code == 200:
        print("✅ Prediction successful:", response.json())
    else:
        print("❌ Prediction failed:", response.status_code, response.text)

print("\nTesting Prediction Endpoint...")
test_prediction()
```
## Monitoring with Prometheus 📊
Prometheus is deployed alongside the application and can be accessed via a web interface. This project uses prometheus_flask_exporter to expose metrics from the Flask application

The Prometheus UI can be accessed here: http://3.235.130.3:9090/


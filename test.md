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

    Checking API Status...
    ✅ API is running: {'message': 'Housing Prediction API is up!'}
    
    Testing Prediction Endpoint...
    ✅ Prediction successful: {'prediction': 721983.1693424652}


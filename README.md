# Housing Prediction Project

## Video Demonstrations
Google Drive Link: https://drive.google.com/drive/folders/1ViCtnuH5xBPZPYHfUDzNVYKBDCdlXLeP?usp=drive_link

## Overview
- This project aims to analyze housing market data to develop predictive models for home price trends. It includes data preprocessing, model training, and deployment using cloud-based services.
- The project is structured to be deployed on AWS or GCP.


## Project Structure
app/  
application.py  # Main application script  
requirements.txt  # Required dependencies  

data/  
sampled_data.csv  # Processed dataset for modeling  

scripts/  
preprocess.py  # Data preprocessing script  
deploy.py  # Deployment automation script  
destroy.py  # Cleanup script for resources  
get_eb_ip.py  # Get Elastic Beanstalk IP script  

README.md  # Project documentation  
.gitignore  # Ignore unnecessary files  

## Setup Instructions
1. **Clone the repository**  
   ```bash
   git clone https://github.com/TNKatneni/MSDS434.git
   cd housing-prediction

2. **Create a virtual environment and install dependencies**
    ```bash  
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r app/requirements.txt

3. **Deploy the application**
    ```bash
    python scripts/deploy.py

4. **Run the application**
    ```bash
    python app/application.py

5. **Destroy the deployed resources**
    ```bash
    python scripts/destroy.py





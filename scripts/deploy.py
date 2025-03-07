"""
scripts/deploy.py
-----------------
Usage:
    python deploy.py
"""

import os
import time
import zipfile
import boto3
import botocore
import paramiko
from botocore.exceptions import ClientError

# ----------------------
# CONFIGURATIONS
# ----------------------
REGION = "us-east-1"
S3_BUCKET_NAME = "tarun-housing-bucket-2025"
LOCAL_TRAINING_DATA = "data/sampled_data.csv"
S3_TRAINING_KEY = "training_data/sampled_data.csv"
SAGEMAKER_ROLE_ARN = "arn:aws:iam::131369287207:role/housingpredictrole"
SAGEMAKER_JOB_NAME_PREFIX = "housing-xgboost-job"
MODEL_NAME = "housing-xgboost-model"
ENDPOINT_CONFIG_NAME = "housing-xgboost-endpoint-config"
ENDPOINT_NAME = "my-housing-endpoint"

EB_APP_NAME = "housing-app-tarun"
EB_ENV_NAME = "housing-env-tarun"
EB_PLATFORM = "Python 3.8 running on 64bit Amazon Linux 2"

# Paths to your app folder
APP_FOLDER = "app"
APP_ZIPFILE = "app_bundle.zip"

# ----------------------
# CREATE S3 BUCKET & UPLOAD
# ----------------------
def create_s3_bucket(bucket_name, region=REGION):
    s3_client = boto3.client("s3", region_name=region)
    try:
        if region == "us-east-1":
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": region},
            )
        print(f"Created bucket: {bucket_name}")
    except ClientError as e:
        if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
            print(f"Bucket {bucket_name} already exists and is owned by you.")
        elif e.response['Error']['Code'] == 'BucketAlreadyExists':
            raise ValueError(f"Bucket name {bucket_name} is already taken.")
        else:
            raise e

def upload_file_to_s3(local_path, bucket_name, s3_key):
    s3_client = boto3.client("s3", region_name=REGION)
    print(f"Uploading {local_path} to s3://{bucket_name}/{s3_key}")
    s3_client.upload_file(local_path, bucket_name, s3_key)
    print("Upload complete.")

# ----------------------
# SAGEMAKER TRAINING
# ----------------------
def create_training_job(bucket_name, s3_data_key, region=REGION, role_arn=SAGEMAKER_ROLE_ARN):
    sm_client = boto3.client("sagemaker", region_name=region)

    # Generate a unique training job name every time
    job_name = f"{SAGEMAKER_JOB_NAME_PREFIX}-{int(time.time())}"
    print(f"🚀 Starting new training job: {job_name}")

    xgboost_container_uri = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.5-1"

    response = sm_client.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            "TrainingImage": xgboost_container_uri,
            "TrainingInputMode": "File"
        },
        RoleArn=role_arn,
        InputDataConfig=[
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{bucket_name}/{s3_data_key}",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "text/csv"
            }
        ],
        OutputDataConfig={
            "S3OutputPath": f"s3://{bucket_name}/output"
        },
        ResourceConfig={
            "InstanceType": "ml.m4.xlarge",
            "InstanceCount": 1,
            "VolumeSizeInGB": 5
        },
        HyperParameters={
            "num_round": "50",
            "objective": "reg:squarederror"
        },
        StoppingCondition={
            "MaxRuntimeInSeconds": 3600
        },
    )
    print(f"✅ Created new training job: {job_name}")
    return job_name  # Return the new training job name

def wait_for_training_job(job_name, region=REGION):
    sm_client = boto3.client("sagemaker", region_name=region)
    print(f"Waiting for training job {job_name} to complete...")
    waiter = sm_client.get_waiter("training_job_completed_or_stopped")
    waiter.wait(TrainingJobName=job_name)
    desc = sm_client.describe_training_job(TrainingJobName=job_name)
    status = desc["TrainingJobStatus"]
    if status == "Failed":
        message = desc["FailureReason"]
        raise Exception(f"Training job failed: {message}")
    elif status == "Stopped":
        raise Exception("Training job was stopped.")
    print("Training job completed successfully.")

# ----------------------
# CREATE SAGEMAKER MODEL & ENDPOINT
# ----------------------
def create_sagemaker_model_and_endpoint(job_name, model_name, endpoint_config_name, endpoint_name, region=REGION):
    sm_client = boto3.client("sagemaker", region_name=region)
    # Get details from the training job
    training_info = sm_client.describe_training_job(TrainingJobName=job_name)
    model_data_url = training_info["ModelArtifacts"]["S3ModelArtifacts"]
    container_image = training_info["AlgorithmSpecification"]["TrainingImage"]
    role_arn = training_info["RoleArn"]

    # Create Model
    sm_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": container_image,
            "ModelDataUrl": model_data_url
        },
        ExecutionRoleArn=role_arn
    )
    print(f"✅ Created model: {model_name}")

    # Create Endpoint Config
    sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InstanceType": "ml.m4.xlarge",
                "InitialInstanceCount": 1
            }
        ]
    )
    print(f"✅ Created endpoint config: {endpoint_config_name}")

    # Create Endpoint
    sm_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name
    )
    print(f"🚀 Creating endpoint: {endpoint_name}")

    # Wait for the endpoint to be InService
    waiter = sm_client.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=endpoint_name)
    print(f"✅ Endpoint {endpoint_name} is now InService!")

# ----------------------
# DEPLOY TO ELASTIC BEANSTALK
# ----------------------
def zip_app(source_folder, zip_name):
    """
    Zip up the contents of source_folder into zip_name.
    """
    zip_path = os.path.join(os.getcwd(), zip_name)  # Ensure full path
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(source_folder):
            for f in files:
                full_path = os.path.join(root, f)
                relative_path = os.path.relpath(full_path, start=source_folder)
                zf.write(full_path, relative_path)
    print(f"Zipped {source_folder} -> {zip_path}")

def get_platform_arn(platform_name):
    """
    Retrieve the platform ARN for Elastic Beanstalk.
    If no ARN is needed, return the platform name directly.
    """
    return "arn:aws:elasticbeanstalk:us-east-1::platform/Python 3.8 running on 64bit Amazon Linux 2/3.7.9"

def ensure_eb_app_exists(eb_client, app_name):
    """
    Ensure that the Elastic Beanstalk application exists.
    If it doesn't, create it.
    """
    response = eb_client.describe_applications(ApplicationNames=[app_name])
    apps = response.get("Applications", [])
    if not apps:
        print(f"❌ EB application '{app_name}' not found. Creating it now...")
        eb_client.create_application(ApplicationName=app_name)
        # Optionally, wait a few seconds for the app to be registered
        time.sleep(5)
        print(f"✅ Created EB application: {app_name}")
    else:
        print(f"✅ EB application '{app_name}' found.")

def deploy_eb_app(app_name, env_name, platform, version_label, bucket_name, app_zip):
    eb_client = boto3.client("elasticbeanstalk", region_name=REGION)
    s3_client = boto3.client("s3", region_name=REGION)

    # Ensure the EB application exists
    ensure_eb_app_exists(eb_client, app_name)

    # Upload app bundle to S3
    s3_key = f"eb-deploy/{app_zip}"
    app_zip_path = os.path.join("scripts", app_zip)  # Correct path in the project root
    print(f"Uploading app zip to s3://{bucket_name}/{s3_key}")
    s3_client.upload_file(app_zip_path, bucket_name, s3_key)

    # Create application version
    print(f"Creating application version '{version_label}' for {app_name}")
    eb_client.create_application_version(
        ApplicationName=app_name,
        VersionLabel=version_label,
        SourceBundle={
            "S3Bucket": bucket_name,
            "S3Key": s3_key
        },
        Process=True  # This tells EB to process the version
    )

    # Wait for application version to be processed
    print(f"⏳ Waiting for application version '{version_label}' to be processed...")
    while True:
        time.sleep(10)
        response = eb_client.describe_application_versions(
            ApplicationName=app_name,
            VersionLabels=[version_label]
        )
        version_status = response["ApplicationVersions"][0]["Status"]
        print(f"   - Status: {version_status}")
        if version_status.upper() == "PROCESSED":
            print(f"✅ Application version '{version_label}' is ready.")
            break
        elif version_status.upper() == "FAILED":
            raise Exception(f"❌ Application version '{version_label}' failed to process.")

    # Create or update the environment
    print(f"Deploying to environment: {env_name}")
    try:
        eb_client.create_environment(
            ApplicationName=app_name,
            EnvironmentName=env_name,
            PlatformArn=get_platform_arn(platform),
            VersionLabel=version_label,
            OptionSettings=[
                {
                    # So your Flask code knows which SageMaker endpoint to call
                    "Namespace": "aws:elasticbeanstalk:application:environment",
                    "OptionName": "SM_ENDPOINT_NAME",
                    "Value": ENDPOINT_NAME
                },
                {
                    # This line associates your custom instance profile
                    "Namespace": "aws:autoscaling:launchconfiguration",
                    "OptionName": "IamInstanceProfile",
                    "Value": "aws-elasticbeanstalk-ec2-role-housing"
                },
                {
                    "Namespace": "aws:autoscaling:launchconfiguration",
                    "OptionName": "EC2KeyName",
                    "Value": "housingpredict"  # Replace with your actual key pair name
                }
            ]
        )
    except ClientError as e:
        if "already exists" in str(e):
            print(f"⚠️ Environment '{env_name}' already exists. Updating with new version.")
            eb_client.update_environment(
                EnvironmentName=env_name,
                VersionLabel=version_label,
                OptionSettings=[
                    {
                        "Namespace": "aws:autoscaling:launchconfiguration",
                        "OptionName": "IamInstanceProfile",
                        "Value": "aws-elasticbeanstalk-ec2-role-housing"
                    }
                ]
            )
        else:
            raise e

    print("✅ Elastic Beanstalk deployment started!")

def wait_for_eb_ready(env_name):
    """
    Waits for the Elastic Beanstalk environment to be 'Ready'.
    """
    eb_client = boto3.client("elasticbeanstalk", region_name=REGION)

    print(f"⏳ Waiting for {env_name} to be 'Ready'...")

    while True:
        response = eb_client.describe_environments(EnvironmentNames=[env_name])
        if not response["Environments"]:
            raise Exception(f"❌ No environment found for {env_name}")

        status = response["Environments"][0]["Status"]
        health = response["Environments"][0]["Health"]

        print(f"   - Status: {status}, Health: {health}")

        if status == "Ready" and health in ["Green", "Ok"]:
            print(f"✅ {env_name} is Ready!")
            break

        time.sleep(30)  # Wait 30 seconds before checking again


def get_eb_instance_ip(env_name):
    """
    Retrieves the public IP address of the EC2 instance running Elastic Beanstalk.
    """
    ec2_client = boto3.client("ec2", region_name=REGION)
    eb_client = boto3.client("elasticbeanstalk", region_name=REGION)

    # Get EB environment information
    env_desc = eb_client.describe_environments(EnvironmentNames=[env_name])
    if not env_desc["Environments"]:
        raise Exception(f"❌ No environment found for {env_name}")

    # Get EC2 instance ID from EB environment
    env_resources = eb_client.describe_environment_resources(EnvironmentId=env_desc["Environments"][0]["EnvironmentId"])
    instance_ids = [i["Id"] for i in env_resources["EnvironmentResources"]["Instances"]]

    if not instance_ids:
        raise Exception(f"❌ No running instances found for {env_name}")

    # Get the public IP of the first instance
    reservations = ec2_client.describe_instances(InstanceIds=instance_ids)
    public_ip = reservations["Reservations"][0]["Instances"][0].get("PublicIpAddress", None)

    if not public_ip:
        raise Exception(f"❌ No public IP found for instances in {env_name}")

    print(f"✅ EB instance public IP: {public_ip}")
    return public_ip

def wait_for_eb_environment(env_name, max_retries=30, delay=10):
    """
    Waits for the EB environment to be in the 'Ready' state before proceeding.
    """
    eb_client = boto3.client("elasticbeanstalk", region_name=REGION)

    print(f"⏳ Waiting for EB environment '{env_name}' to be ready...")
    
    for i in range(max_retries):
        env_desc = eb_client.describe_environments(EnvironmentNames=[env_name])
        if env_desc["Environments"]:
            status = env_desc["Environments"][0]["Status"]
            print(f"   - Attempt {i+1}: EB status is '{status}'")
            if status.lower() == "ready":
                print(f"✅ EB environment '{env_name}' is ready!")
                return
        time.sleep(delay)
    
    raise Exception(f"❌ EB environment '{env_name}' did not reach 'Ready' state after {max_retries * delay} seconds")

def install_dependencies(instance_ip, ssh_key_path):
    """
    Installs required dependencies on the EC2 instance.
    """
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(instance_ip, username="ec2-user", key_filename=ssh_key_path)

    print("📦 Installing required Python dependencies...")
    commands = [
        "sudo python3 -m pip install --upgrade pip",
        "sudo python3 -m pip install flask boto3 prometheus_client prometheus_flask_exporter"
    ]

    for cmd in commands:
        ssh.exec_command(cmd)
        time.sleep(5)  # Allow some time for installations

    print("✅ Dependencies installed!")
    ssh.close()


def setup_monitoring(env_name):
    """
    Automatically configures Prometheus & Grafana on the EB instance.
    """
    wait_for_eb_environment(env_name)  # Ensure EB environment is ready
    instance_ip = get_eb_instance_ip(env_name)  # Retrieve instance IP
    ssh_key_path = "/Users/tarunkatneni/Desktop/Cloud Computing/housingpredict.pem"

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(instance_ip, username="ec2-user", key_filename=ssh_key_path)

    print("📦 Installing Prometheus & Grafana...")
    commands = [
        # Install Grafana
        "wget https://dl.grafana.com/oss/release/grafana-10.0.3-1.x86_64.rpm",
        "sudo yum localinstall -y grafana-10.0.3-1.x86_64.rpm",
        "sudo systemctl enable grafana-server",
        "sudo systemctl start grafana-server",
        
        # Install Prometheus
        "wget https://github.com/prometheus/prometheus/releases/download/v2.51.2/prometheus-2.51.2.linux-amd64.tar.gz",
        "tar -xvf prometheus-2.51.2.linux-amd64.tar.gz",
        "sudo mv prometheus-2.51.2.linux-amd64 /opt/prometheus",
        "sudo ln -s /opt/prometheus/prometheus /usr/local/bin/"
    ]

    for cmd in commands:
        ssh.exec_command(cmd)
        time.sleep(5)

    # Ensure Prometheus directory exists
    ssh.exec_command("sudo mkdir -p /etc/prometheus")

    # Create Prometheus config file
    prometheus_config = """\
global:
  scrape_interval: 10s

scrape_configs:
  - job_name: "flask_app"
    static_configs:
      - targets: ["localhost:8080"]
  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]
"""
    ssh.exec_command(f"echo '{prometheus_config}' | sudo tee /etc/prometheus/prometheus.yml")
    
    # Ensure Prometheus service is restarted or started if not running
    ssh.exec_command("sudo systemctl restart prometheus || sudo systemctl start prometheus")

    print("✅ Monitoring setup complete!")
    ssh.close()


# ----------------------
# MAIN DEPLOY SEQUENCE
# ----------------------
def main():
    # 1. Create S3 bucket and upload training data
    create_s3_bucket(S3_BUCKET_NAME, REGION)
    upload_file_to_s3(LOCAL_TRAINING_DATA, S3_BUCKET_NAME, S3_TRAINING_KEY)

    # 2. Always create a new SageMaker training job
    training_job_name = create_training_job(S3_BUCKET_NAME, S3_TRAINING_KEY)
    wait_for_training_job(training_job_name)

    # 2.5 Create SageMaker model and endpoint automatically
    create_sagemaker_model_and_endpoint(training_job_name, MODEL_NAME, ENDPOINT_CONFIG_NAME, ENDPOINT_NAME)

    # 3. Deploy Flask app to Elastic Beanstalk
    project_root = os.getcwd()  
    os.chdir(APP_FOLDER)  

    zip_app(".", os.path.join(project_root, "scripts", APP_ZIPFILE))  

    os.chdir(project_root)  

    VERSION_LABEL = f"v-{int(time.time())}"
    deploy_eb_app(EB_APP_NAME, EB_ENV_NAME, EB_PLATFORM, VERSION_LABEL, S3_BUCKET_NAME, APP_ZIPFILE)

    # **📌 Wait for EB environment to be ready**
    wait_for_eb_environment(EB_ENV_NAME)

    # **📌 Install Python dependencies AFTER EB is ready**
    instance_ip = get_eb_instance_ip(EB_ENV_NAME)
    install_dependencies(instance_ip, "/Users/tarunkatneni/Desktop/Cloud Computing/housingpredict.pem")

    # **📌 Now set up Prometheus & Grafana**
    setup_monitoring(EB_ENV_NAME)

    print("🚀 Deployment complete with monitoring enabled!")


if __name__ == "__main__":
    main()

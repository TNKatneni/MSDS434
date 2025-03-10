import time
import boto3
from botocore.exceptions import ClientError

REGION = "us-east-1"
S3_BUCKET_NAME = "tarun-housing-bucket-2025"
ENDPOINT_NAME = "my-housing-endpoint"
ENDPOINT_CONFIG_NAME = "housing-xgboost-endpoint-config"
MODEL_NAME = "housing-xgboost-model"
SAGEMAKER_JOB_NAME = "housing-xgboost-job"
EB_APP_NAME = "housing-app-tarun"
EB_ENV_NAME = "housing-env-tarun"

def delete_eb_env_and_app(app_name, env_name):
    eb_client = boto3.client("elasticbeanstalk", region_name=REGION)

    # Terminate environment
    print(f"Terminating EB environment: {env_name}")
    try:
        eb_client.terminate_environment(EnvironmentName=env_name)
    except ClientError as e:
        print(f"Could not terminate environment: {e}")

    # Wait for environment to terminate
    while True:
        time.sleep(20)
        try:
            resp = eb_client.describe_environments(EnvironmentNames=[env_name])
            envs = resp["Environments"]
            if not envs:
                print("Environment already gone.")
                break
            status = envs[0].get("Status", "")
            print(f"Environment status: {status}")
            if status in ["Terminated", "Terminating"]:
                if status == "Terminated":
                    print("Environment is terminated.")
                    break
        except ClientError:
            break

    # Delete application versions & application
    print(f"Deleting EB application: {app_name}")
    try:
        versions_resp = eb_client.describe_application_versions(ApplicationName=app_name)
        versions = versions_resp["ApplicationVersions"]
        for v in versions:
            label = v["VersionLabel"]
            print(f"Deleting application version: {label}")
            try:
                eb_client.delete_application_version(
                    ApplicationName=app_name,
                    VersionLabel=label,
                    DeleteSourceBundle=True
                )
            except ClientError as e:
                print(f"Error deleting version {label}: {e}")

        eb_client.delete_application(ApplicationName=app_name, TerminateEnvByForce=True)
        print("Application deleted.")
    except ClientError as e:
        print(f"Error deleting application: {e}")

def delete_sagemaker_endpoint(endpoint_name):
    sm_client = boto3.client("sagemaker", region_name=REGION)
    print(f"Deleting endpoint {endpoint_name}...")
    try:
        sm_client.delete_endpoint(EndpointName=endpoint_name)
    except ClientError as e:
        print(f"Error deleting endpoint: {e}")
        return

    while True:
        time.sleep(10)
        try:
            sm_client.describe_endpoint(EndpointName=endpoint_name)
        except ClientError:
            print("Endpoint deleted.")
            break

def delete_endpoint_config(config_name):
    sm_client = boto3.client("sagemaker", region_name=REGION)
    print(f"Deleting endpoint config {config_name}...")
    try:
        sm_client.delete_endpoint_config(EndpointConfigName=config_name)
        print("Endpoint config deleted.")
    except ClientError as e:
        print(f"Error deleting endpoint config: {e}")

def delete_model(model_name):
    sm_client = boto3.client("sagemaker", region_name=REGION)
    print(f"Deleting model {model_name}...")
    try:
        sm_client.delete_model(ModelName=model_name)
        print("Model deleted.")
    except ClientError as e:
        print(f"Error deleting model: {e}")

def stop_training_job(job_name):
    """
    Attempts to stop a SageMaker training job if it's still in progress.
    AWS does not allow deleting completed jobs.
    """
    sm_client = boto3.client("sagemaker", region_name=REGION)
    
    try:
        desc = sm_client.describe_training_job(TrainingJobName=job_name)
        status = desc["TrainingJobStatus"]
        
        if status in ["InProgress", "Stopping"]:
            print(f"Stopping training job {job_name}...")
            sm_client.stop_training_job(TrainingJobName=job_name)
            print("Training job stopped.")
        else:
            print(f"Training job {job_name} is already completed or stopped.")
    
    except ClientError as e:
        print(f"Could not stop training job: {e}")

def empty_and_delete_bucket(bucket_name):
    s3 = boto3.resource("s3", region_name=REGION)
    bucket = s3.Bucket(bucket_name)
    print(f"Deleting all objects from bucket: {bucket_name}")
    try:
        bucket.objects.all().delete()
        print("Emptied bucket.")
    except ClientError as e:
        print(f"Error emptying bucket: {e}")

    print(f"Deleting bucket {bucket_name}...")
    try:
        bucket.delete()
        print("Bucket deleted.")
    except ClientError as e:
        print(f"Error deleting bucket: {e}")

def main():
    # 1. Stop the SageMaker training job if it exists
    stop_training_job(SAGEMAKER_JOB_NAME)

    # 2. Delete SageMaker endpoint, config, model
    delete_sagemaker_endpoint(ENDPOINT_NAME)
    delete_endpoint_config(ENDPOINT_CONFIG_NAME)
    delete_model(MODEL_NAME)

    # 3. Terminate EB environment & application
    delete_eb_env_and_app(EB_APP_NAME, EB_ENV_NAME)

    # 4. Empty and delete S3 bucket
    empty_and_delete_bucket(S3_BUCKET_NAME)

    print("All AWS resources have been cleaned up.")

if __name__ == "__main__":
    main()

import boto3

REGION = 'us-east-1'
EB_ENV_NAME = 'housing-env-tarun'

def get_eb_instance_ip(env_name):
    ec2_client = boto3.client('ec2', region_name=REGION)
    eb_client = boto3.client('elasticbeanstalk', region_name=REGION)

    try:
        env_desc = eb_client.describe_environments(EnvironmentNames=[env_name])
        if not env_desc['Environments']:
            raise Exception(f'❌ No environment found for {env_name}')

        env_resources = eb_client.describe_environment_resources(EnvironmentId=env_desc['Environments'][0]['EnvironmentId'])
        instance_ids = [i['Id'] for i in env_resources['EnvironmentResources']['Instances']]

        if not instance_ids:
            raise Exception(f'❌ No running instances found for {env_name}')

        reservations = ec2_client.describe_instances(InstanceIds=instance_ids)
        public_ip = reservations['Reservations'][0]['Instances'][0].get('PublicIpAddress')

        if not public_ip:
            raise Exception(f'❌ No public IP found for instance {instance_ids[0]}')

        print(f'✅ EB instance public IP: {public_ip}')
    except Exception as e:
        print(f'⚠️ Error: {str(e)}')

get_eb_instance_ip(EB_ENV_NAME)


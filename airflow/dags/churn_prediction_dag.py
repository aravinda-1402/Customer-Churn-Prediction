from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.amazon.aws.operators.sagemaker import SageMakerTrainingOperator
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import boto3

# Function to download the file from S3 using Boto3
def download_file_from_s3():
    s3 = boto3.client('s3')
    s3.download_file('your-bucket-name', 'your-dataset.csv', '/tmp/churn.csv')

default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
    'retries': 1,
}

with DAG('churn_prediction_pipeline', default_args=default_args, schedule_interval='@daily') as dag:
    
    start = DummyOperator(task_id='start')

    download_dataset = PythonOperator(
        task_id='download_dataset',
        python_callable=download_file_from_s3
    )

    train_model = SageMakerTrainingOperator(
        task_id='train_model',
        config={
            'TrainingJobName': 'your-training-job-name',
            'AlgorithmSpecification': {
                'TrainingImage': 'your-docker-image-uri',
                'TrainingInputMode': 'File'
            },
            'RoleArn': 'your-role-arn',
            'InputDataConfig': [
                {
                    'ChannelName': 'train',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': 's3://your-bucket-name/your-dataset.csv',
                            'S3DataDistributionType': 'FullyReplicated',
                        }
                    },
                    'ContentType': 'text/csv',
                    'InputMode': 'File',
                }
            ],
            'OutputDataConfig': {
                'S3OutputPath': 's3://your-bucket-name/model-output/'
            },
            'ResourceConfig': {
                'InstanceType': 'ml.m5.large',
                'InstanceCount': 1,
                'VolumeSizeInGB': 10,
            },
            'StoppingCondition': {
                'MaxRuntimeInSeconds': 86400
            }
        },
        aws_conn_id='aws_default'  # Ensure the AWS connection includes the region
    )

    deploy_flask_app = BashOperator(
        task_id='deploy_flask_app',
        bash_command=(
            "docker run -d "
            "--name flask_app_container "
            "-p 5000:5000 "
            "-e AWS_ACCESS_KEY_ID=your_access_key "
            "-e AWS_SECRET_ACCESS_KEY=your_secret_key "
            "-e AWS_DEFAULT_REGION=us-east-1 "
            "your-docker-image-uri"
        )
    )

    end = DummyOperator(task_id='end')

    start >> download_dataset >> train_model >> deploy_flask_app >> end

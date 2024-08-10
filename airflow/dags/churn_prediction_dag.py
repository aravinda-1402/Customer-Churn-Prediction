from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.amazon.aws.operators.sagemaker import SageMakerTrainingOperator
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.transfers.local_to_s3 import LocalFilesystemToS3Operator
from airflow.utils.dates import days_ago
import boto3

# Function to download the file from S3 using Boto3
def download_file_from_s3():
    s3 = boto3.client('s3')
    s3.download_file('your-bucket-name', 'your-object-key.csv', '/tmp/churn.csv')  # Redacted

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
            'TrainingJobName': 'churn-prediction-job',
            'AlgorithmSpecification': {
                'TrainingImage': 'your-training-image-url',  # Redacted
                'TrainingInputMode': 'File'
            },
            'RoleArn': 'arn:aws:iam::your-iam-role-arn',  # Redacted
            'InputDataConfig': [
                {
                    'ChannelName': 'train',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': 's3://your-bucket-name/your-object-key.csv',  # Redacted
                            'S3DataDistributionType': 'FullyReplicated',
                        }
                    },
                    'ContentType': 'text/csv',
                    'InputMode': 'File',
                }
            ],
            'OutputDataConfig': {
                'S3OutputPath': 's3://your-bucket-name/model-output/'  # Redacted
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


    end = DummyOperator(task_id='end')

    start >> download_dataset >> train_model >> end

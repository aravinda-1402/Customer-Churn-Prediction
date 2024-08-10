# Use the official Airflow image as the base image
FROM apache/airflow:2.5.0

# Install the Amazon provider package and other Python dependencies
RUN pip install --no-cache-dir boto3 sagemaker pandas scikit-learn 'apache-airflow[amazon]'

# Set the working directory inside the container
WORKDIR /app

# Copy the model training script into the container
COPY train.py .

# Define the command to run the model training script
ENTRYPOINT ["python", "train.py"]

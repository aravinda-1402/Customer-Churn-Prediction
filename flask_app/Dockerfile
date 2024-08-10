# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Flask and other dependencies
RUN pip install --no-cache-dir Flask==2.0.3 Werkzeug==2.0.3 boto3==1.24.28 joblib==1.4.0 pandas==1.3.3 scikit-learn==0.24.2

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=app.py

# Run Flask when the container launches
CMD ["flask", "run", "--host=0.0.0.0"]


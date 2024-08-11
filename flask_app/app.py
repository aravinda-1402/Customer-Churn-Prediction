from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import boto3
from io import BytesIO
import logging

app = Flask(__name__)

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load the model, scaler, and label encoders from S3
bucket_name = 'your-bucket-name'  # Redacted bucket name
s3 = boto3.client('s3')

try:
    logging.info("Loading model from S3...")
    
    # Load model
    model_buffer = BytesIO()
    s3.download_fileobj(bucket_name, 'model-output/model.joblib', model_buffer)
    model_buffer.seek(0)
    model = joblib.load(model_buffer)

    # Load scaler
    scaler_buffer = BytesIO()
    s3.download_fileobj(bucket_name, 'model-output/scaler.joblib', scaler_buffer)
    scaler_buffer.seek(0)
    scaler = joblib.load(scaler_buffer)

    # Load label encoders
    encoders_buffer = BytesIO()
    s3.download_fileobj(bucket_name, 'model-output/label_encoders.joblib', encoders_buffer)
    encoders_buffer.seek(0)
    label_encoders = joblib.load(encoders_buffer)

    logging.info("Model and preprocessors loaded successfully from S3.")

except Exception as e:
    logging.error(f"Error loading model or preprocessors: {str(e)}")
    raise

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route for the web UI
@app.route('/predict', methods=['POST'])
def predict():
    logging.info("predict route was called")
    
    # Extract the form data
    input_data = request.form.to_dict()

    # Convert the input data into the format required by the model
    features = []
    for feature, value in input_data.items():
        if feature in label_encoders:  # If the feature is categorical
            value = label_encoders[feature].transform([value])[0]
        features.append(float(value))

    # Convert to numpy array and reshape
    features = np.array(features).reshape(1, -1)
    
    # Scale the numerical features
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    features[:, :3] = scaler.transform(features[:, :3])

    # Make the prediction
    prediction = model.predict(features)

    # Return the result
    if prediction[0] == 1:
        result = "This customer is likely to churn."
    else:
        result = "This customer is not likely to churn."

    return render_template('index.html', prediction_text=result)

# Define the new API route
@app.route('/api/predict', methods=['POST'])
def api_predict():
    logging.info("api_predict route was called")
    
    try:
        # Extract JSON data from the request
        data = request.get_json(force=True)

        # Convert the JSON data into the format required by the model
        features = []
        for feature, value in data.items():
            if feature in label_encoders:  # If the feature is categorical
                value = label_encoders[feature].transform([value])[0]
            features.append(float(value))

        # Convert to numpy array and reshape
        features = np.array(features).reshape(1, -1)
        
        # Scale the numerical features
        num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        features[:, :3] = scaler.transform(features[:, :3])

        # Make the prediction
        prediction = model.predict(features)

        # Return the result as JSON
        result = {'prediction': int(prediction[0])}
        return jsonify(result)

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 400

# Log all registered routes to ensure the route is available
for rule in app.url_map.iter_rules():
    logging.info(f"Endpoint: {rule.endpoint}, Route: {rule}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import boto3
from io import BytesIO, StringIO

# Initialize S3 client
s3 = boto3.client('s3')

# Load the data directly from S3 into a Pandas DataFrame
bucket_name = 'your-bucket-name'  # Replace with your actual bucket name
object_key = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

# Fetch the CSV file from S3 as a string
csv_obj = s3.get_object(Bucket=bucket_name, Key=object_key)
body = csv_obj['Body'].read().decode('utf-8')
data = pd.read_csv(StringIO(body))

# Data preprocessing
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data.drop(labels=data[data['tenure'] == 0].index, axis=0, inplace=True)
data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace=True)
data['SeniorCitizen'] = data['SeniorCitizen'].map({0: "No", 1: "Yes"})

# Save LabelEncoders for categorical features
label_encoders = {}

def object_to_int(dataframe_series, column_name):
    if dataframe_series.dtype == 'object':
        le = LabelEncoder()
        dataframe_series = le.fit_transform(dataframe_series)
        label_encoders[column_name] = le
    return dataframe_series

data = data.apply(lambda x: object_to_int(x, x.name))

X = data.drop(columns=['customerID', 'Churn'])
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_cols = ["tenure", 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the trained model to an in-memory buffer
model_buffer = BytesIO()
joblib.dump(model, model_buffer)
model_buffer.seek(0)  # Move to the beginning of the buffer

# Save the scaler to an in-memory buffer
scaler_buffer = BytesIO()
joblib.dump(scaler, scaler_buffer)
scaler_buffer.seek(0)  # Move to the beginning of the buffer

# Save the label encoders to an in-memory buffer
encoders_buffer = BytesIO()
joblib.dump(label_encoders, encoders_buffer)
encoders_buffer.seek(0)  # Move to the beginning of the buffer

# Upload the model, scaler, and label encoders directly to S3 from the buffers
s3.upload_fileobj(model_buffer, bucket_name, 'model-output/model.joblib')
s3.upload_fileobj(scaler_buffer, bucket_name, 'model-output/scaler.joblib')
s3.upload_fileobj(encoders_buffer, bucket_name, 'model-output/label_encoders.joblib')

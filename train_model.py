import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import kagglehub  # For downloading datasets from Kaggle

# Downloading  dataset from Kaggle
dir_path = kagglehub.dataset_download("shubham14p3/boston-mlops-assignment")  # Replace with  Kaggle dataset path

# Loading  dataset from  downloaded directory
csv_path = f"{dir_path}/boston.csv"  # Adjusting if  file name or path differs
boston_data = pd.read_csv(csv_path)

# Preparing features and target
X = boston_data.drop(columns=["MV"])
y = boston_data["MV"]

# Train  Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Scale  features
scaler = StandardScaler()
scaler.fit(X)

# Save  trained model and scaler
joblib.dump(model, "model/trained_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("Model and scaler have been saved successfully!")

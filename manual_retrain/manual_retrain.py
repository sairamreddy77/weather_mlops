import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.lightgbm
from datetime import datetime
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
TEMP_MODEL_NAME = os.getenv("TEMP_MODEL_NAME")
HUM_MODEL_NAME = os.getenv("HUM_MODEL_NAME")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client["weather_db"]
collection = db["hourly_weather_data"]

print("📥 Loading data from MongoDB...")
docs = list(collection.find())
df = pd.DataFrame(docs)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime")

# Feature Engineering
print("⚙️ Performing feature engineering...")
df["hour"] = df["datetime"].dt.hour
df["dayofweek"] = df["datetime"].dt.dayofweek
df["month"] = df["datetime"].dt.month
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# Lag features
N = 6
for i in range(1, N + 1):
    df[f"temp_lag_{i}"] = df["temperature"].shift(i)
    df[f"humidity_lag_{i}"] = df["humidity"].shift(i)

df.dropna(inplace=True)

features = [f"temp_lag_{i}" for i in range(1, N + 1)] + \
           [f"humidity_lag_{i}" for i in range(1, N + 1)] + \
           ["hour", "dayofweek", "month", "hour_sin", "hour_cos", "month_sin", "month_cos"]

X = df[features]
y_temp = df["temperature"]
y_humidity = df["humidity"]

X_train, X_test, y_temp_train, y_temp_test = train_test_split(X, y_temp, test_size=0.2, random_state=42)
_, _, y_hum_train, y_hum_test = train_test_split(X, y_humidity, test_size=0.2, random_state=42)


# Setup MLflow
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN
os.environ["MLFLOW_TRACKING_TOKEN"] = os.getenv("MLFLOW_TRACKING_TOKEN") 
os.environ["DATABRICKS_HOST"] = os.getenv("DATABRICKS_HOST")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_registry_uri("databricks-uc")

print("🚀 Training and logging models to MLflow...")

# Temperature Model
with mlflow.start_run(run_name="TempModel"):
    model_temp = LGBMRegressor()
    model_temp.fit(X_train, y_temp_train)

    mlflow.log_params(model_temp.get_params())
    mlflow.log_metric("r2_temp", model_temp.score(X_test, y_temp_test))
    mlflow.lightgbm.log_model(model_temp, artifact_path="model", registered_model_name=TEMP_MODEL_NAME)

    # Update alias
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(TEMP_MODEL_NAME, stages=["None"])[0].version
    client.set_registered_model_alias(TEMP_MODEL_NAME, "prod", latest_version)

# Humidity Model
with mlflow.start_run(run_name="HumidityModel"):
    model_hum = LGBMRegressor()
    model_hum.fit(X_train, y_hum_train)

    mlflow.log_params(model_hum.get_params())
    mlflow.log_metric("r2_humidity", model_hum.score(X_test, y_hum_test))
    mlflow.lightgbm.log_model(model_hum, artifact_path="model", registered_model_name=HUM_MODEL_NAME)

    # Update alias
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(HUM_MODEL_NAME, stages=["None"])[0].version
    client.set_registered_model_alias(HUM_MODEL_NAME, "prod", latest_version)

print("✅ Retraining and alias update complete.")

import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.lightgbm
from mlflow.models.signature import infer_signature
from datetime import datetime
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
TEMP_MODEL_NAME = os.getenv("TEMP_MODEL_NAME")
HUMIDITY_MODEL_NAME = os.getenv("HUMIDITY_MODEL_NAME")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

# Setup MLflow URIs
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN
os.environ["MLFLOW_TRACKING_TOKEN"] = DATABRICKS_TOKEN
os.environ["DATABRICKS_HOST"] = os.getenv("DATABRICKS_HOST")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Users/sairambreddy@gmail.com/weather-prediction")

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

print("📊 Training feature columns:", X.columns.tolist())
print("🧼 Nulls in training data:\n", X.isnull().sum())
print("🔢 Shape of training data:", X.shape)

# ----------------------------
# ✅ Train Temperature Model
# ----------------------------
with mlflow.start_run(run_name="TempModel") as run_temp:
    model_temp = LGBMRegressor()
    model_temp.fit(X_train, y_temp_train)

    signature_temp = infer_signature(X_test, model_temp.predict(X_test))
    input_example_temp = X_test.head(1)

    mlflow.log_params(model_temp.get_params())
    mlflow.log_metric("r2_temp", model_temp.score(X_test, y_temp_test))
    
    mlflow.lightgbm.log_model(
        model_temp,
        artifact_path="model",
        registered_model_name=TEMP_MODEL_NAME,
        input_example=input_example_temp,
        signature=signature_temp
    )

# ✅ Update alias for latest version
client = mlflow.MlflowClient()
all_versions_temp = client.search_model_versions(f"name='{TEMP_MODEL_NAME}'")
latest_version_temp = sorted(all_versions_temp, key=lambda v: int(v.version))[-1]
client.set_registered_model_alias(TEMP_MODEL_NAME, alias="prod", version=latest_version_temp.version)

# ----------------------------
# ✅ Train Humidity Model
# ----------------------------
with mlflow.start_run(run_name="HumidityModel") as run_hum:
    model_hum = LGBMRegressor()
    model_hum.fit(X_train, y_hum_train)

    signature_hum = infer_signature(X_test, model_hum.predict(X_test))
    input_example_hum = X_test.head(1)

    mlflow.log_params(model_hum.get_params())
    mlflow.log_metric("r2_humidity", model_hum.score(X_test, y_hum_test))
    
    mlflow.lightgbm.log_model(
        model_hum,
        artifact_path="model",
        registered_model_name=HUMIDITY_MODEL_NAME,
        input_example=input_example_hum,
        signature=signature_hum
    )

# ✅ Update alias for latest version
all_versions_hum = client.search_model_versions(f"name='{HUMIDITY_MODEL_NAME}'")
latest_version_hum = sorted(all_versions_hum, key=lambda v: int(v.version))[-1]
client.set_registered_model_alias(HUMIDITY_MODEL_NAME, alias="prod", version=latest_version_hum.version)

print("✅ Retraining and alias update complete.")


# retrain_and_log.py
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from pymongo import MongoClient
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_TOKEN"] = os.getenv("MLFLOW_TRACKING_TOKEN") 

os.environ["DATABRICKS_HOST"] = os.getenv("DATABRICKS_HOST")
os.environ["DATABRICKS_TOKEN"] = os.getenv("DATABRICKS_TOKEN")

# Databricks Tracking URI
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
# Tell MLflow to use Unity Catalog registry
mlflow.set_registry_uri("databricks-uc")

mlflow.set_experiment("/Users/sairambreddy@gmail.com/weather-prediction")

# MongoDB connection
client = MongoClient(os.getenv("MONGODB_URI"))
db = client["weather_db"]
collection = db["hourly_weather_data"]

# Load and prepare data
def load_data():
    data = pd.DataFrame(list(collection.find()))
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)
    return data

data = load_data()
features = [col for col in data.columns if col not in ["_id", "temperature", "humidity", "datetime"]]
X = data[features]
y_temp = data["temperature"]
y_hum = data["humidity"]
X_train, X_test, y_temp_train, y_temp_test = train_test_split(X, y_temp, test_size=0.2, random_state=42)
_, _, y_hum_train, y_hum_test = train_test_split(X, y_hum, test_size=0.2, random_state=42)

# Train models
temp_model = lgb.LGBMRegressor()
hum_model = lgb.LGBMRegressor()

temp_model.fit(X_train, y_temp_train)
hum_model.fit(X_train, y_hum_train)

# Log to MLflow
with mlflow.start_run(run_name="TempModel_Retrain"):
    mlflow.log_params(temp_model.get_params())
    mlflow.lightgbm.log_model(temp_model, artifact_path="model", registered_model_name="mlops_catalog.ml_models.temp_model")
    mlflow.log_metric("r2", r2_score(y_temp_test, temp_model.predict(X_test)))
    mlflow.log_metric("mse", mean_squared_error(y_temp_test, temp_model.predict(X_test)))

with mlflow.start_run(run_name="HumModel_Retrain"):
    mlflow.log_params(hum_model.get_params())
    mlflow.lightgbm.log_model(hum_model, artifact_path="model", registered_model_name="mlops_catalog.ml_models.humidity_model")
    mlflow.log_metric("r2", r2_score(y_hum_test, hum_model.predict(X_test)))
    mlflow.log_metric("mse", mean_squared_error(y_hum_test, hum_model.predict(X_test)))



# i have an idea i think we have to foresee.
# we have limited resources.
# we cant keep running the servers continuously 
# and since u brought up the concept of scheduling it got me thinking ,
# how bout we run the servers for limited time and then infer and store for the whole day at once   
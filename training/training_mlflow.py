import mlflow
import mlflow.lightgbm
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
#dapi5c4e2824530a7a897c2178a63b77d29e

os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_TOKEN"] = os.getenv("MLFLOW_TRACKING_TOKEN") 

os.environ["DATABRICKS_HOST"] = os.getenv("DATABRICKS_HOST")
os.environ["DATABRICKS_TOKEN"] = os.getenv("DATABRICKS_TOKEN")

# Databricks Tracking URI
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
# Tell MLflow to use Unity Catalog registry
mlflow.set_registry_uri("databricks-uc")

# Optional: print to verify
print("MLflow Tracking URI:", mlflow.get_tracking_uri())
# Optional: print to verify
print("MLflow Registry URI:", mlflow.get_registry_uri())

# Set experiment (must exist in Databricks MLflow workspace)
mlflow.set_experiment("/Users/sairambreddy@gmail.com/weather-prediction")

# Load your cleaned dataset with datetime
df = pd.read_csv("hyderabad_weather_2yrs.csv", parse_dates=["datetime"])
df = df.sort_values("datetime")

# Add time-based features
df["hour"] = df["datetime"].dt.hour
df["dayofweek"] = df["datetime"].dt.dayofweek
df["month"] = df["datetime"].dt.month
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)


# Create lag features (e.g., last 6 hours)
N = 6
for i in range(1, N + 1):
    df[f"temp_lag_{i}"] = df["temp_c"].shift(i)
    df[f"humidity_lag_{i}"] = df["humidity"].shift(i)

df.dropna(inplace=True)

# Features
features = [f"temp_lag_{i}" for i in range(1, N + 1)] + \
           [f"humidity_lag_{i}" for i in range(1, N + 1)] + \
           ["hour", "dayofweek", "month",
            "hour_sin", "hour_cos", "month_sin", "month_cos"]

# Train models
X = df[features]
y_temp = df["temp_c"]
y_humidity = df["humidity"]

X_train, X_test, y_temp_train, y_temp_test = train_test_split(X, y_temp, test_size=0.2, random_state=42)
_, _, y_hum_train, y_hum_test = train_test_split(X, y_humidity, test_size=0.2, random_state=42)


# 🚀 Train and log Temperature model
with mlflow.start_run(run_name="TemperatureModel"):
    model_temp = lgb.LGBMRegressor()
    model_temp.fit(X_train, y_temp_train)

    # Predict
    y_pred_temp = model_temp.predict(X_test)

    # Metrics
    r2 = r2_score(y_temp_test, y_pred_temp)
    rmse = np.sqrt(mean_squared_error(y_temp_test, y_pred_temp))
    mae = mean_absolute_error(y_temp_test, y_pred_temp)

    # Log all
    mlflow.log_params(model_temp.get_params())
    mlflow.log_metric("temp_r2", r2)
    mlflow.log_metric("temp_rmse", rmse)
    mlflow.log_metric("temp_mae", mae)
    
    # ✅ Save model
    mlflow.lightgbm.log_model(model_temp,
                            artifact_path="temperature_model", 
                            input_example=X_test.head(1).to_dict(orient="records"),
                            registered_model_name="mlops_catalog.ml_models.temp_model")
    print("✅ Temperature model logged to MLflow")

# 🚀 Train and log Humidity model
with mlflow.start_run(run_name="HumidityModel"):
    model_hum = lgb.LGBMRegressor()
    model_hum.fit(X_train, y_hum_train)

    # Predict
    y_pred_hum = model_hum.predict(X_test)

    # Metrics
    r2 = r2_score(y_hum_test, y_pred_hum)
    rmse = np.sqrt(mean_squared_error(y_hum_test, y_pred_hum))
    mae = mean_absolute_error(y_hum_test, y_pred_hum)

    # Log all
    mlflow.log_params(model_hum.get_params())
    mlflow.log_metric("humidity_r2", r2)
    mlflow.log_metric("humidity_rmse", rmse)
    mlflow.log_metric("humidity_mae", mae)

    # ✅ Save model
    mlflow.lightgbm.log_model(model_hum,
                            artifact_path="humidity_model", 
                            input_example=X_test.head(1).to_dict(orient="records"),
                            registered_model_name="mlops_catalog.ml_models.humidity_model")
    print("✅ Humidity model logged to MLflow")





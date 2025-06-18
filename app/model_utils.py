import numpy as np
import pandas as pd
from pymongo import MongoClient
from datetime import datetime, timedelta
# from joblib import load
# from lightgbm import Booster
import mlflow.pyfunc
from dotenv import load_dotenv
import os


# temp_model = Booster(model_file="models/lgb_temp_model.txt")
# humidity_model = Booster(model_file="models/lgb_humidity_model.txt")

load_dotenv()

os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_TOKEN"] = os.getenv("MLFLOW_TRACKING_TOKEN") 

os.environ["DATABRICKS_HOST"] = os.getenv("DATABRICKS_HOST")
os.environ["DATABRICKS_TOKEN"] = os.getenv("DATABRICKS_TOKEN")


mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_registry_uri("databricks-uc")

# Load them models from the model registry
temp_model = mlflow.pyfunc.load_model(f"models:/{os.getenv('TEMP_MODEL_NAME')}/1")
hum_model = mlflow.pyfunc.load_model(f"models:/{os.getenv('HUMIDITY_MODEL_NAME')}/1")


MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["weather_db"]
collection = db["hourly_weather_data"]

def get_recent_features():
    # Get the most recent 6 hours from MongoDB
    cursor = collection.find().sort("datetime", -1).limit(6)
    docs = list(cursor)[::-1]  # Reverse to get ascending order

    if len(docs) < 6:
        raise ValueError("Not enough data to compute lag features.")

    # Create DataFrame
    df = pd.DataFrame(docs)

    def get_ist_now():
        return datetime.utcnow() + timedelta(hours=5.5)

    # Feature engineering
    now = get_ist_now()
    hour = now.hour
    dayofweek = now.weekday()
    month = now.month

    feature_dict = {
        **{f"temp_lag_{i+1}": float(df.iloc[-(i+1)]["temperature"]) for i in range(6)},
        **{f"humidity_lag_{i+1}": float(df.iloc[-(i+1)]["humidity"]) for i in range(6)},
        "hour": int(hour),
        "dayofweek": int(dayofweek),
        "month": int(month),
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        "month_sin": float(np.sin(2 * np.pi * month / 12)),
        "month_cos": float(np.cos(2 * np.pi * month / 12)),
    }

    return pd.DataFrame([feature_dict])

def predict_temp_humidity():
    X = get_recent_features()

    temp = temp_model.predict(X)[0]
    humidity = hum_model.predict(X)[0]

    return round(temp, 1), round(humidity, 1)

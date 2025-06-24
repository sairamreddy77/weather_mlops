from flask import Blueprint, jsonify
from datetime import datetime, timedelta
import requests
from mongo_utils import weather_col, prediction_col
# from model_utils import predict_temp_humidity
from weather_fetch_and_store import fetch_and_store_24hr_data
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import mlflow

load_dotenv()

API_KEY = os.getenv("WEATHER_API_KEY")
CITY = "Hyderabad"

os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_TOKEN"] = os.getenv("MLFLOW_TRACKING_TOKEN") 

os.environ["DATABRICKS_HOST"] = os.getenv("DATABRICKS_HOST")
os.environ["DATABRICKS_TOKEN"] = os.getenv("DATABRICKS_TOKEN")


mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_registry_uri("databricks-uc")

# Load them models from the model registry
temp_model = mlflow.pyfunc.load_model(f"models:/{os.getenv('TEMP_MODEL_NAME')}/1")
hum_model = mlflow.pyfunc.load_model(f"models:/{os.getenv('HUMIDITY_MODEL_NAME')}/1")


weather_blueprint = Blueprint("weather", __name__)

# Helper to collect and store raw weather data
def fetch_and_store_raw_weather():
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={CITY}"
    response = requests.get(url)
    if response.status_code != 200:
        return None

    data = response.json()
    dt = data["location"]["localtime"]
    hour = int(data["location"]["localtime"].split(" ")[1].split(":")[0])
    month = int(data["location"]["localtime"].split("-")[1])
    temperature = data["current"]["temp_c"]
    humidity = data["current"]["humidity"]

    document = {
        "datetime": dt,
        "hour": hour,
        "month": month,
        "temperature": temperature,
        "humidity": humidity,
    }

    # weather_col.update_one({"datetime": dt}, {"$set": document}, upsert=True)
    return document

@weather_blueprint.route("/current", methods=["GET"])
def get_current_weather():
    doc = fetch_and_store_raw_weather()
    if doc:
        return jsonify({"status": "success", "data": doc})
    return jsonify({"status": "error", "message": "Failed to fetch data"})

@weather_blueprint.route("/forecast", methods=["GET"])
def forecast_next_hours():
    #add the previous 24 hours data if not already present
    fetch_and_store_24hr_data()

    def get_ist_now():
        return datetime.utcnow() + timedelta(hours=5.5)

    try:
        now = get_ist_now()
        base_time = now.replace(minute=0, second=0, microsecond=0)

        # Step 1: Get latest 6 hourly data from MongoDB for lag features
        past_cursor = prediction_col.database["hourly_weather_data"].find().sort("datetime", -1).limit(6)
        latest_data = list(past_cursor)[::-1]  # reverse to chronological order

        if len(latest_data) < 6:
            return jsonify({"status": "error", "message": "Not enough data for forecasting"})

        predictions = []

        # Step 2: Loop over next 6 hours
        for i in range(6):
            future_time = base_time + timedelta(hours=i + 1)

            # Prepare lag features
            df = pd.DataFrame(latest_data[-6:])  # use the latest 6 entries
            features = {
                **{f"temp_lag_{j+1}": float(df.iloc[-(j+1)]["temperature"]) for j in range(6)},
                **{f"humidity_lag_{j+1}": float(df.iloc[-(j+1)]["humidity"]) for j in range(6)},
                "hour": int(future_time.hour),
                "dayofweek": int(future_time.weekday()),
                "month": int(future_time.month),
                "hour_sin": float(np.sin(2 * np.pi * future_time.hour / 24)),
                "hour_cos": float(np.cos(2 * np.pi * future_time.hour / 24)),
                "month_sin": float(np.sin(2 * np.pi * future_time.month / 12)),
                "month_cos": float(np.cos(2 * np.pi * future_time.month / 12)),
            }

            X = pd.DataFrame([features])
            temp = round(temp_model.predict(X)[0], 1)
            hum = round(hum_model.predict(X)[0], 1)

            pred_doc = {
                "datetime": future_time.strftime("%Y-%m-%d %H:00"),
                "hour": future_time.hour,
                "month": future_time.month,
                "predicted_temperature": temp,
                "predicted_humidity": hum
            }

            # Store prediction in MongoDB
            prediction_col.update_one({"datetime": pred_doc["datetime"]}, {"$set": pred_doc}, upsert=True)
            predictions.append(pred_doc)

            # Step 3: Add the prediction to the lag window
            latest_data.append({
                "datetime": future_time.strftime("%Y-%m-%d %H:00"),
                "temperature": temp,
                "humidity": hum
            })

        return jsonify({"status": "success", "predictions": predictions})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

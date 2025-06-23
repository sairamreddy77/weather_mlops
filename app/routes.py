from flask import Blueprint, jsonify
from datetime import datetime, timedelta
import requests
from mongo_utils import weather_col, prediction_col
from model_utils import predict_temp_humidity
from weather_fetch_and_store import fetch_and_store_24hr_data
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("WEATHER_API_KEY")
CITY = "Hyderabad"

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

    now = get_ist_now()
    predictions = []
    for i in range(1, 7):  # next 6 hours
        future = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=i)
        hour = future.hour
        month = future.month
        dt_str = future.strftime("%Y-%m-%d %H:00")

        
        try:
            temp, humidity = predict_temp_humidity()
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)})

        pred_doc = {    
            "datetime": dt_str,
            "hour": hour,
            "month": month,
            "predicted_temperature": temp,
            "predicted_humidity": humidity
        }

        prediction_col.update_one({"datetime": dt_str}, {"$set": pred_doc}, upsert=True)
        predictions.append(pred_doc)

    return jsonify({"status": "success", "predictions": predictions})

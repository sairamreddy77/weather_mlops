import requests
import pandas as pd
from pymongo import MongoClient
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()

# Configuration
API_KEY = os.getenv("WEATHER_API_KEY")
MONGO_URI =os.getenv("MONGO_URI")
DATABASE_NAME = "weather_db"
COLLECTION_NAME = "hourly_weather_data"
LOCATION = "Hyderabad"

# Setup MongoDB client
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

def fetch_24_hour_data():
    def get_ist_now():
        return datetime.utcnow() + timedelta(hours=5.5)
    
    now = get_ist_now()
    yesterday = now - timedelta(days=1)

    dates_to_fetch = [yesterday.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")]
    all_data = []

    for date_str in dates_to_fetch:
        url = f"http://api.weatherapi.com/v1/history.json?key={API_KEY}&q={LOCATION}&dt={date_str}"
        response = requests.get(url)
        if response.status_code == 200:
            forecast_data = response.json().get("forecast", {}).get("forecastday", [])
            if forecast_data:
                all_data.extend(forecast_data[0]["hour"])

    # Filter last 24 hours only
    time_limit = now - timedelta(hours=24)
    result_data = []
    for entry in all_data:
        dt = pd.to_datetime(entry["time"])
        if time_limit <= dt <= now:
            result_data.append({
                "datetime": dt.strftime("%Y-%m-%d %H:%M"),
                "hour": dt.hour,
                "month": dt.month,
                "temperature": entry["temp_c"],
                "humidity": entry["humidity"]
            })

    return result_data

def insert_if_not_exists(data):
    inserted_count = 0
    for doc in data:
        if not collection.find_one({"datetime": doc["datetime"]}):
            collection.insert_one(doc)
            inserted_count += 1
    return inserted_count

def fetch_and_store_24hr_data():
    weather_data = fetch_24_hour_data()
    count = insert_if_not_exists(weather_data)
    print(f"{count} new documents inserted.")
    return weather_data[-1] if weather_data else None  # Return latest hour record

import requests
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["weather_db"]
live_col = db["hourly_weather_data"]

API_KEY = os.getenv("WEATHER_API_KEY")

def fetch_past_24hr_weather(include_extra_lag_hours=False):
    base_url = "http://api.weatherapi.com/v1/history.json"
    location = "Hyderabad"

    def get_ist_now():
        return datetime.utcnow() + timedelta(hours=5.5)

    now = get_ist_now()
    start = now - timedelta(hours=30 if include_extra_lag_hours else 24)

    data = []
    current_hour = start

    while current_hour <= now:
        date_str = current_hour.strftime("%Y-%m-%d")
        url = f"{base_url}?key={API_KEY}&q={location}&dt={date_str}"
        response = requests.get(url)
        hour_data = response.json()["forecast"]["forecastday"][0]["hour"]

        for record in hour_data:
            dt = datetime.strptime(record["time"], "%Y-%m-%d %H:%M")
            if start <= dt <= now:
                doc = {
                    "datetime": dt.strftime("%Y-%m-%d %H:00"),
                    "temperature": record["temp_c"],
                    "humidity": record["humidity"],
                    "hour": dt.hour,
                    "month": dt.month
                }
                data.append(doc)
                # ✅ Store in MongoDB
                live_col.update_one({"datetime": doc["datetime"]}, {"$set": doc}, upsert=True)

        current_hour += timedelta(days=1)

    return data

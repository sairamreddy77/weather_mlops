import requests
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from pymongo import MongoClient
load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGO_URI)
db = client["weather_db"]
live_col = db["hourly_weather_data"]



API_KEY = os.getenv("WEATHER_API_KEY")

def fetch_past_24hr_weather():
    base_url = "http://api.weatherapi.com/v1/history.json"
    location = "Hyderabad"

    def get_ist_now():
        return datetime.utcnow() + timedelta(hours=5.5)
    
    now = get_ist_now()
    yesterday = now - timedelta(days=1)
    data = []

    for hour in range(24):
        time_point = yesterday + timedelta(hours=hour)
        date_str = time_point.strftime("%Y-%m-%d")
        url = f"{base_url}?key={API_KEY}&q={location}&dt={date_str}"
        response = requests.get(url)
        hour_data = response.json()["forecast"]["forecastday"][0]["hour"]
        for record in hour_data:
            dt = datetime.strptime(record["time"], "%Y-%m-%d %H:%M")
            if dt >= yesterday:
                doc={"datetime": dt.strftime("%Y-%m-%d %H:00"),
                    "temperature": record["temp_c"],
                    "humidity": record["humidity"],
                    "hour": dt.hour,
                    "month": dt.month
                    
                }
                data.append(doc)

        live_col.update_one({"datetime": doc["datetime"]}, {"$set": doc}, upsert=True)  # Store in MongoDB here
    return data

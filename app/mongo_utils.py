from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
# Use your actual MongoDB Atlas URI
MONGO_URI =os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["weather_db"]

weather_col = db["hourly_weather_data"]
prediction_col = db["weather_predictions"]

# Ensure index for deduplication
prediction_col.create_index("datetime", unique=True)
weather_col.create_index("datetime", unique=True)

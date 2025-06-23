from pymongo import MongoClient
import os
import json
from dotenv import load_dotenv
load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGO_URI)
db = client["weather_db"]
prediction_col = db["daily_predictions"]

def store_predictions(predictions):
    for pred in predictions:
        prediction_col.update_one(
            {"datetime": pred["datetime"]},
            {"$set": pred},
            upsert=True
        )

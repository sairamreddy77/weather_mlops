from pymongo import MongoClient
import os
import json
from dotenv import load_dotenv
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["weather_db"]
prediction_col = db["daily_predictions"]

def store_predictions(predictions):
    if not predictions:
        print("❌ No predictions to store.")
        return

    for pred in predictions:
        try:
            result = prediction_col.update_one(
                {"datetime": pred["datetime"]},
                {"$set": pred},
                upsert=True
            )
            print(f"✅ Stored prediction for {pred['datetime']}")
        except Exception as e:
            print(f"❌ Failed to store prediction: {e}")
    print("💾 All predictions stored successfully.")

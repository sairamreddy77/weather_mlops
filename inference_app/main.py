from fetch_weather import fetch_past_24hr_weather
from predict import predict_next_6_hours
from store import store_predictions

import os
from dotenv import load_dotenv  
load_dotenv()

if __name__ == "__main__":
    try:
        print("🔐 Connecting to Mongo URI:", os.getenv("MONGO_URI"))

        print("📦 Fetching past 24 hours data...")
        data = fetch_past_24hr_weather()
        print("🔮 Predicting next 6 hours...")
        predictions = predict_next_6_hours(data)
        print("💾 Storing predictions in MongoDB...")
        print("Predictions:", predictions)
        store_predictions(predictions)
        print("✅ Done.")
    except Exception as e:
        print(f"❌ Job failed: {e}")
        raise
    

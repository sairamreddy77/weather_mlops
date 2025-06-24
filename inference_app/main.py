from fetch_weather import fetch_past_24hr_weather
from predict import predict_hourly_over_day
from store import store_predictions

import os
from dotenv import load_dotenv  
load_dotenv()

if __name__ == "__main__":
    try:
        print("🔐 Connecting to Mongo URI:", os.getenv("MONGO_URI"))

        print("📦 Fetching past 30 hours data (24 + 6 lags)...")
        data = fetch_past_24hr_weather(include_extra_lag_hours=True)
        print("🔮 Predicting 24 hourly forecasts...")
        predictions = predict_hourly_over_day(data)

        print("💾 Storing predictions in MongoDB...")
        # print("Predictions:", predictions)
        store_predictions(predictions)
        print("✅ Done.")

        
    except Exception as e:
        print(f"❌ Job failed: {e}")
        raise
    

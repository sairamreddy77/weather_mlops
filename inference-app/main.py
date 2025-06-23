from fetch_weather import fetch_past_24hr_weather
from predict import predict_next_6_hours
from store import store_predictions

if __name__ == "__main__":
    print("📦 Fetching past 24 hours data...")
    data = fetch_past_24hr_weather()
    print("🔮 Predicting next 6 hours...")
    predictions = predict_next_6_hours(data)
    print("💾 Storing predictions in MongoDB...")
    store_predictions(predictions)
    print("✅ Done.")

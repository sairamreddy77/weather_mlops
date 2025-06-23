import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from model_utils import get_models

def create_features(latest_data):
    df = pd.DataFrame(latest_data[-6:])  # Last 6 hrs
    def get_ist_now():
        return datetime.utcnow() + timedelta(hours=5.5)
    
    now = get_ist_now()

    features = {
        **{f"temp_lag_{i+1}": df.iloc[-(i+1)]["temperature"] for i in range(6)},
        **{f"humidity_lag_{i+1}": df.iloc[-(i+1)]["humidity"] for i in range(6)},
        "hour": now.hour,
        "dayofweek": now.weekday(),
        "month": now.month,
        "hour_sin": np.sin(2 * np.pi * now.hour / 24),
        "hour_cos": np.cos(2 * np.pi * now.hour / 24),
        "month_sin": np.sin(2 * np.pi * now.month / 12),
        "month_cos": np.cos(2 * np.pi * now.month / 12),
    }
    return pd.DataFrame([features])

def predict_next_6_hours(latest_data):
    temp_model, hum_model = get_models()
    predictions = []

    def get_ist_now():
        return datetime.utcnow() + timedelta(hours=5.5)
    
    now = get_ist_now()

    base_time = now.replace(minute=0, second=0, microsecond=0)
    for i in range(6):
        # Calculate future time
        future_time = base_time + timedelta(hours=i+1)
        X = create_features(latest_data)
        temp = round(temp_model.predict(X)[0], 1)
        hum = round(hum_model.predict(X)[0], 1)
        predictions.append({
            "datetime": future_time.strftime("%Y-%m-%d %H:00"),
            "predicted_temperature": temp,
            "predicted_humidity": hum,
            "hour": future_time.hour,
            "month": future_time.month
        })
        # Simulate data shift for lag feature updates
        latest_data.append({
            "datetime": future_time.strftime("%Y-%m-%d %H:00"),
            "temperature": temp,
            "humidity": hum
        })
    return predictions

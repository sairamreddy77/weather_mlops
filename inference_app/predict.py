import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from model_utils import get_models

def predict_hourly_over_day(all_data):
    temp_model, hum_model = get_models()
    predictions = []

    # Sort by datetime
    df = pd.DataFrame(all_data)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    for i in range(6, 30):  # Start at index 6 to have 6 lag hours
        past_window = df.iloc[i - 6:i]
        predict_time = df.iloc[i]["datetime"]

        features = {
            **{f"temp_lag_{j+1}": float(past_window.iloc[-(j+1)]["temperature"]) for j in range(6)},
            **{f"humidity_lag_{j+1}": float(past_window.iloc[-(j+1)]["humidity"]) for j in range(6)},
            "hour": int(predict_time.hour),
            "dayofweek": int(predict_time.weekday()),
            "month": int(predict_time.month),
            "hour_sin": float(np.sin(2 * np.pi * predict_time.hour / 24)),
            "hour_cos": float(np.cos(2 * np.pi * predict_time.hour / 24)),
            "month_sin": float(np.sin(2 * np.pi * predict_time.month / 12)),
            "month_cos": float(np.cos(2 * np.pi * predict_time.month / 12)),
        }

        X = pd.DataFrame([features])
        temp = round(temp_model.predict(X)[0], 1)
        hum = round(hum_model.predict(X)[0], 1)

        predictions.append({
            "datetime": predict_time.strftime("%Y-%m-%d %H:00"),
            "predicted_temperature": temp,
            "predicted_humidity": hum,
            "hour": predict_time.hour,
            "month": predict_time.month
        })
    
    print(len(predictions))

    return predictions


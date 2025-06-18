import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import os

# Load your cleaned dataset with datetime
df = pd.read_csv("hyderabad_weather_2yrs.csv", parse_dates=["datetime"])
df = df.sort_values("datetime")

# Add time-based features
df["hour"] = df["datetime"].dt.hour
df["dayofweek"] = df["datetime"].dt.dayofweek
df["month"] = df["datetime"].dt.month
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# Create lag features (e.g., last 6 hours)
N = 6
for i in range(1, N + 1):
    df[f"temp_lag_{i}"] = df["temp_c"].shift(i)
    df[f"humidity_lag_{i}"] = df["humidity"].shift(i)

df.dropna(inplace=True)

# Features
features = [f"temp_lag_{i}" for i in range(1, N + 1)] + \
           [f"humidity_lag_{i}" for i in range(1, N + 1)] + \
           ["hour", "dayofweek", "month",
            "hour_sin", "hour_cos", "month_sin", "month_cos"]

# Train models
X = df[features]
y_temp = df["temp_c"]
y_humidity = df["humidity"]

X_train, X_test, y_temp_train, y_temp_test = train_test_split(X, y_temp, test_size=0.2, random_state=42)
_, _, y_hum_train, y_hum_test = train_test_split(X, y_humidity, test_size=0.2, random_state=42)

model_temp = lgb.LGBMRegressor()
model_temp.fit(X_train, y_temp_train)

model_hum = lgb.LGBMRegressor()
model_hum.fit(X_train, y_hum_train)

# ✅ Save boosters instead of full sklearn wrapper
os.makedirs("models", exist_ok=True)
model_temp.booster_.save_model("models/lgb_temp_model.txt")
model_hum.booster_.save_model("models/lgb_humidity_model.txt")

print("✅ Models trained and saved as Booster format (safe for Docker)")

import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta

API_KEY = "b4b37638262b4386b4663424251306"
LOCATION = "Hyderabad"
START_DATE = datetime(2024, 6, 12)
END_DATE = datetime(2025, 6, 12)
CSV_FILE = "C:/Users/SAIRAM REDDY/ml/weather_mlops/hyderabad_weather_2yrs.csv"

# Resume logic
def get_last_date():
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        if "datetime" in df.columns and not df["datetime"].isnull().all():
            df["datetime"] = pd.to_datetime(df["datetime"], errors='coerce')
            last_valid = df["datetime"].dropna().max()
            if pd.notnull(last_valid):
                return last_valid.date() + timedelta(days=1)

    return START_DATE

# Prepare CSV if not present
if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=["datetime", "temp_c", "humidity", "chance_of_rain", "chance_of_snow"]).to_csv(CSV_FILE, index=False)

def fetch_day_weather(date):
    url = f"http://api.weatherapi.com/v1/history.json?key={API_KEY}&q={LOCATION}&dt={date.strftime('%Y-%m-%d')}"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"[{date}] Error: {response.status_code} - {response.text}")
            return []
        data = response.json()
        hourly = data["forecast"]["forecastday"][0]["hour"]
        daily = data["forecast"]["forecastday"][0]["day"]

        result = []
        for hour_data in hourly:
            result.append({
                "datetime": hour_data["time"],
                "temp_c": hour_data["temp_c"],
                "humidity": hour_data["humidity"],
                "chance_of_rain": daily.get("daily_chance_of_rain", 0),
                "chance_of_snow": daily.get("daily_chance_of_snow", 0)
            })
        return result

    except Exception as e:
        print(f"[{date}] Exception occurred: {e}")
        return []

def main():
    current_date = get_last_date()
    print(f"Starting from: {current_date}")

    all_data = []

    while current_date <= END_DATE:
        print(f"Fetching data for: {current_date}")
        day_data = fetch_day_weather(current_date)
        if day_data:
            df = pd.DataFrame(day_data)
            df.to_csv(CSV_FILE, mode='a', header=False, index=False)
            print(f"[{current_date}] ✔ Saved {len(day_data)} rows.")
        else:
            print(f"[{current_date}] ❌ No data fetched.")

        current_date += timedelta(days=1)
        time.sleep(1)  # Respect free API rate limits

if __name__ == "__main__":
    main()

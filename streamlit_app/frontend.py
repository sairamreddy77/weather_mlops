import streamlit as st
import requests
import pandas as pd

# BASE_URL = "http://flask-app:5000"  # Change if hosted elsewhere
BASE_URL = "http://localhost:5000"  # Change if hosted elsewhere

st.title("🌦️ Weather Forecast Dashboard - Hyderabad")

# --- Section: Current Weather ---
st.header("Current Weather (Live)")

if st.button("Fetch Current Weather"):
    resp = requests.get(f"{BASE_URL}/current")
    if resp.status_code == 200:
        data = resp.json().get("data", {})
        st.success("Live weather fetched successfully!")
        st.write(f"**Time**: {data['datetime']}")
        st.write(f"**Temperature**: {data['temperature']}°C")
        st.write(f"**Humidity**: {data['humidity']}%")
    else:
        st.error("Failed to fetch live weather.")

# --- Section: Prediction ---
st.header("🔮 Next 6-Hour Forecast")

if st.button("Get Forecast"):
    resp = requests.get(f"{BASE_URL}/forecast")

    try:
        data = resp.json()
        # st.write("Parsed JSON:", data)
    except Exception as e:
        st.error(f"Failed to parse JSON: {e}")
        st.stop()

    forecast = data.get("predictions", [])
    if not forecast:
        st.error("No forecast data received.")
        st.stop()

    df = pd.DataFrame(forecast)
    # st.write("Forecast DataFrame:", df)

    if "datetime" not in df.columns:
        st.error(f"Expected 'datetime' column not in forecast data: {df.columns}")
        st.stop()

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(by="datetime")
    st.dataframe(df[["datetime", "predicted_temperature", "predicted_humidity"]].set_index("datetime"))

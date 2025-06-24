from dotenv import load_dotenv
import os

load_dotenv()
print("MONGODB_URI:", os.getenv("MONGO_URI"))
print("WEATHER_API_KEY:", os.getenv("WEATHER_API_KEY"))

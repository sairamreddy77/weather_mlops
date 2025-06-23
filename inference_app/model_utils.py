import mlflow.pyfunc
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_TOKEN"] = os.getenv("MLFLOW_TRACKING_TOKEN")
os.environ["DATABRICKS_HOST"] = os.getenv("DATABRICKS_HOST")
os.environ["DATABRICKS_TOKEN"] = os.getenv("DATABRICKS_TOKEN")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_registry_uri("databricks-uc")

# Load models from the model registry
temp_model = mlflow.pyfunc.load_model(f"models:/{os.getenv('TEMP_MODEL_NAME')}/1")
hum_model = mlflow.pyfunc.load_model(f"models:/{os.getenv('HUMIDITY_MODEL_NAME')}/1")

def get_models():
    return temp_model, hum_model

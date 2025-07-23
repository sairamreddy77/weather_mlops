# 🌦️ Weather Forecasting MLOps System

This project is an end-to-end MLOps pipeline designed to forecast hourly **temperature** and **humidity** in Hyderabad using real-time weather data. The system incorporates **model versioning**, **CI/CD**, **containerization**, **scheduled inference**, and **retraining workflows**, simulating a real-world production environment.

---

## 🚀 Features

- **Hourly Weather Prediction**  
  Predicts temperature and humidity using a LightGBM model trained on historical hourly weather data.

- **Daily Scheduled Inference (Azure)**  
  Automatically fetches past 24 hours of data, performs inference for the next 6 hours, and stores predictions in MongoDB.

- **Model Versioning with MLflow**  
  Models are tracked and stored on Databricks MLflow, supporting easy rollback and version comparison.

- **Containerized Architecture with Docker**  
  All components (training, inference, retraining, app) are containerized for reproducibility and deployment.

- **CI/CD Pipeline (GitHub Actions)**  
  Automated testing and container deployment to Azure Container Apps.

- **Data Drift Detection & Monthly Retraining**  
  Live data is continuously monitored; models are retrained monthly with newly collected data to adapt to changing patterns.

- **Monitoring with Prometheus & Grafana**  
  Tracks inference latency, API health, and model performance (planned setup).

---

## 🛠️ Tech Stack

| Component       | Tool/Service                  |
|----------------|-------------------------------|
| Programming    | Python, Pandas, LightGBM       |
| Model Tracking | MLflow (Databricks)            |
| Workflow       | Azure Container Apps, Azure Functions |
| Data Storage   | MongoDB Atlas                  |
| DevOps         | Docker, GitHub Actions         |
| Monitoring     | Prometheus, Grafana (planned)  |
| Frontend       | Streamlit                      |
| Backend        | Flask API                      |

---

## 🔄 Pipeline Overview

    +-----------------+
    | WeatherAPI.com  |
    +--------+--------+
             |
             v
     [Live Data Fetching]
             |
             v
    [MongoDB Atlas Storage]
             |
     +-------+--------+
     |   Inference     |
     | (Scheduled Job) |
     +-------+--------+
             |
     +-------v--------+
     |   Predictions   |
     |  MongoDB Output |
     +----------------+
             |
     +-------v--------+
     |   Streamlit     |
     |   Frontend UI   |
     +----------------+



     
---

## 📅 Workflows

### 🧠 Model Training
- Trained using sliding window on 1 year of hourly data.
- Targets: Temperature, Humidity
- Tracked on MLflow (Databricks)
- Saved as TorchScript/LightGBM binary

### 📈 Inference (Scheduled Daily)
- Runs every 24 hours via Azure Job
- Predicts next 6 hours based on previous 24
- Stores output in MongoDB for frontend

### 🔁 Retraining
- Triggered monthly (via Azure Container App job)
- Uses latest live data from MongoDB
- New model pushed to MLflow and tagged

### 📊 Monitoring (Planned)
- Logs metrics like inference time, error rate, model drift
- Prometheus scrapes app endpoints
- Grafana dashboards visualize metrics

---

## ✅ Getting Started (Dev)

```bash
git clone https://github.com/your-username/weather-mlops.git
cd weather-mlops

# Build and run containers
docker-compose up --build


# Hazara Division Air Quality Intelligence (AQI) System

> **Advanced Machine Learning Pipeline for Real-Time Air Quality Forecasting**

> *10Pearls Internship Project | Production-Grade Architecture | Automated CI/CD*

---

## Live Production Application

The system is deployed live in production:
* **Production URL**: [https://hazara-aqi.onrender.com](https://hazara-aqi.onrender.com)

---

## Project Overview

The **Hazara Division AQI Prediction System** is an end-to-end, automated machine learning platform designed to forecast the Air Quality Index (AQI) for the **Hazara Division, Khyber Pakhtunkhwa (KPK), Pakistan**, for the next 72 hours.

This system features a **self-correcting ML pipeline** that automatically fetches new data, trains multiple algorithms (from statistical models to deep learning), and dynamically promotes the best-performing model to production. The system covers **6 districts** with a modular, scalable architecture.

### Covered Districts

| District | Latitude | Longitude | Description |
| --- | --- | --- | --- |
| **Abbottabad** (HQ) | 34.1463 | 73.2117 | Divisional headquarters, most urbanized |
| Mansehra | 34.3302 | 73.1968 | Second largest city in the division |
| Haripur | 33.9942 | 72.9333 | Industrial zone near Taxila |
| Battagram | 34.6837 | 73.0261 | Northern mountainous district |
| Upper Kohistan | 35.2097 | 73.3456 | Remote high-altitude region |
| Torghar | 34.6300 | 72.9400 | Newly formed district |

## Key Features

* **Multi-District Monitoring**: Real-time AQI data for 6 districts across Hazara Division.
* **Adaptive Machine Learning**: Trains 6 models (including deep learning) and auto-selects the best by RMSE.
* **Deep Learning (LSTM)**: TensorFlow/Keras LSTM neural network for time-series AQI forecasting.
* **Feature Store**: Hopsworks integration for centralized feature management and model registry.
* **Fully Automated Pipeline**: GitHub Actions manage hourly data ingestion and daily model retraining.
* **FastAPI REST API**: Production-grade JSON API endpoints for integration with any frontend or service.
* **Interactive Dashboard**: Beautiful dark-themed HTML dashboard with Plotly charts, 3-day outlook, and CSV export.
* **Health Alerts**: Automatic color-coded health advisory banners when AQI exceeds unhealthy thresholds.
* **Explainability**: SHAP-based feature importance analysis notebooks.

---

## Project Structure

```text
Hazara-AQI/
├── app/                            # Web Application (FastAPI)
│   ├── main.py                     # FastAPI backend + REST API endpoints
│   └── templates/
│       └── dashboard.html          # Jinja2 HTML template (dark theme)
│
├── src/                            # Backend Logic & Pipelines
│   ├── __init__.py                 # Package initializer
│   ├── data_ingestion.py           # Open-Meteo API data fetching

│   ├── preprocessing.py            # Data cleaning & outlier removal
│   ├── feature_engineering.py      # Lag, rolling average, temporal features
│   ├── feature_store.py            # Hopsworks Feature Store integration
│   ├── modeling.py                 # ML model training & inference
│   └── lstm_model.py              # TensorFlow LSTM deep learning model
│
├── models/                         # Model Artifacts
│   ├── model.pkl                   # Best ML model (scikit-learn)
│   ├── features.pkl                # Feature names for consistency
│   ├── lstm_model.keras            # LSTM deep learning model (TensorFlow)
│   ├── lstm_scaler.pkl             # MinMaxScaler for LSTM
│   └── training_results.csv        # 6-model leaderboard
│
├── data/                           # Local Data Store
│   ├── raw_hazara_aqi.csv          # Raw API data (26,640 records)
│   ├── cleaned_hazara_aqi.csv      # Cleaned & preprocessed
│   └── features_hazara_aqi.csv     # Feature-engineered training data
│
├── notebooks/                      # Research & Experiments
│   ├── EDA.ipynb                   # Exploratory Data Analysis
│   └── Shap_analysis.ipynb         # SHAP Feature Importance
│
├── .github/workflows/              # CI/CD Automation
│   ├── daily_retrain.yml           # Daily model retraining (midnight)
│   └── hourly_data_update.yml      # Hourly data collection
│
├── run_pipeline.py                 # End-to-end pipeline (Steps 1-6)
├── Dockerfile                      # Container configuration
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
└── .gitignore                      # Git exclusions
```

---

## Model Performance

The system trains 6 models (statistical + ensemble + deep learning) and selects the best:

| Model Type | RMSE | MAE | R² Score | Status |
| --- | --- | --- | --- | --- |
| **Linear Regression** | **4.6713** | **2.8675** | **0.9823** | **🏆 Active** |
| Ridge Regression | 4.6713 | 2.8676 | 0.9823 | Candidate |
| Gradient Boosting | 5.9069 | 3.2007 | 0.9716 | Candidate |
| XGBoost | 6.2753 | 3.3111 | 0.9680 | Candidate |
| Random Forest | 7.3534 | 3.8424 | 0.9561 | Candidate |
| LSTM (Deep Learning) | 8.4066 | 5.8862 | 0.9426 | Candidate |

---

## Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Hazara-AQI.git
cd Hazara-AQI
```

### 2. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Complete Pipeline

```bash
python run_pipeline.py
```

This runs all 6 steps:
1. **Data Collection** — Fetches 6 months of hourly data from Open-Meteo API
2. **Data Cleaning** — Handles NaN values, clips outliers (1st-99th percentile)
3. **Feature Engineering** — Generates 13 features (temporal + rolling + lag + derived)
4. **ML Model Training** — Trains Linear, Ridge, Random Forest, Gradient Boosting, XGBoost
5. **LSTM Training** — Trains a TensorFlow/Keras LSTM neural network
6. **Hopsworks Feature Store** — Uploads features and registers model (if API key set)

### 4. Launch the FastAPI Dashboard

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000` in your browser.

### 5. Configure Environment (Optional)

```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```env
HOPSWORKS_API_KEY=your_hopsworks_api_key
HOPSWORKS_PROJECT=HazaraAQI
```

---

## REST API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/` | Dashboard HTML page |
| GET | `/api/current?district=Abbottabad` | Current AQI + category |
| GET | `/api/forecast?district=Abbottabad` | 72-hour forecast (JSON) |
| GET | `/api/history?district=Abbottabad` | Historical data (7 days) |
| GET | `/api/districts` | List all 6 districts |
| GET | `/api/models` | Model performance leaderboard |

---

## Technology Stack

| Component | Technology |
| --- | --- |
| **Data Source** | Open-Meteo Air Quality API |
| **Feature Store** | Hopsworks (free tier) |
| **ML Models** | Scikit-learn, XGBoost |
| **Deep Learning** | TensorFlow / Keras (LSTM) |
| **Web Framework** | FastAPI + Jinja2 |
| **Visualization** | Plotly.js |
| **CI/CD** | GitHub Actions |
| **Containerization** | Docker |
| **Interpretability** | SHAP |
| **Language** | Python 3.10 |

---

## License

This project is open-source and available under the MIT License.

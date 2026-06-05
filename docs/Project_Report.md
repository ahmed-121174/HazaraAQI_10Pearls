# Hazara Division AQI Prediction & Monitoring System
## Comprehensive Technical Project Report

---

## 1. Executive Summary
This report documents the design, implementation, and deployment of the **Hazara Division Air Quality Index (AQI) Prediction & Monitoring System**. Developed for Abbottabad, Haripur, Mansehra, Battagram, Upper Kohistan, and Torghar districts in KPK, Pakistan, this system addresses a key environmental challenge: monitoring and forecasting localized air pollution in complex mountainous terrains.

The project implements a complete machine learning lifecycle (MLOps) system:
1. **Data Pipeline**: Automated hourly ingestion of meteorological and air quality pollutant data from the Open-Meteo API.
2. **Feature Engineering**: Deriving 13 dynamic historical features (lags, rolling stats, rate of change, temporal cycles).
3. **Multi-Model Training**: Training a suite of 6 models ranging from statistical (Linear & Ridge Regression) to ensemble and boosting (Random Forest, Gradient Boosting, XGBoost), and deep learning (univariate LSTM).
4. **Explainable AI**: Calculating Shapley Additive exPlanations (SHAP) to interpret feature contributions.
5. **Interactive Dashboard**: A FastAPI-based, responsive, dark-themed interface rendering real-time metrics, Plotly charts, API documentation, full 72-hour CSV downloads, and dynamic color-coded health alerts.
6. **Automation & CI/CD**: Background system management via a bash controller (`run.sh`) and production CI/CD workflows for automated retraining and ingestion.

---

## 2. Geographical Context & Atmospheric Dynamics
Hazara Division covers an area of approximately 17,197 km² in northern KPK, characterized by steep valleys, high mountain ranges (Himalayas/Hindukush), and expanding urban hubs. 

Unlike flat coastal areas like Karachi, Hazara's air quality is governed by unique atmospheric processes:
* **Temperature Inversions**: Cold mountain air sinks into valleys overnight, trapping vehicular and industrial emissions (especially in urban basins like Abbottabad and Haripur) close to the ground.
* **Transboundary Pollution**: Seasonal agricultural burning and dust transport accumulate in deep valleys.
* **Topographical Barriers**: Narrow valley channels prevent rapid wind dispersion, resulting in persistent air quality patterns.

### District Coordinates Configured
The data pipeline monitors coordinates for the following locations:
* **Abbottabad (HQ)**: 34.1463° N, 73.2117° E
* **Mansehra**: 34.3302° N, 73.1968° E
* **Haripur**: 33.9942° N, 72.9333° E
* **Battagram**: 34.6837° N, 73.0261° E
* **Upper Kohistan**: 35.2097° N, 73.3456° E
* **Torghar**: 34.6300° N, 72.9400° E

---

## 3. Data Pipeline & Preprocessing
The system fetches hourly data from the **Open-Meteo Air Quality API**, capturing:
* Particulate Matter: $\text{PM}_{2.5}$ and $\text{PM}_{10}$ ($\mu\text{g/m}^3$)
* Gaseous Pollutants: Carbon Monoxide ($\text{CO}$), Nitrogen Dioxide ($\text{NO}_2$), Sulphur Dioxide ($\text{SO}_2$), Ozone ($\text{O}_3$)
* Overall Index: United States AQI (US AQI)

### Preprocessing & Cleaning
* **Outliers**: Clipped features at the 1st and 99th percentiles to mitigate sensor malfunctions or extreme anomalies without discarding historical sequences.
* **Imputation**: Missing values are imputed using forward-fill (FFill) to propagate the last known air state, followed by backward-fill (BFill) for any remaining edge cases.

---

## 4. Feature Engineering Design
To forecast AQI 72 hours into the future, the model relies on historical indicators:
1. **Temporal Context**: `hour` (diurnal traffic patterns), `day_of_week` (weekend reductions), and `month` (seasonal fuel burning and winter heating).
2. **Lag Features**: `lag_1h_aqi`, `lag_2h_aqi`, and `lag_24h_aqi` capture short-term persistence and diurnal cycles.
3. **Moving Averages**: 6-hour and 24-hour rolling averages (`pm10_rolling_24h`, `pm2_5_rolling_24h`, `us_aqi_rolling_24h`, etc.) smooth out hourly volatility to reflect ambient background accumulation.
4. **Derivative Features**: `aqi_change_rate` (first-order difference) captures the velocity of pollution buildup or dispersal.

---

## 5. Model Architecture & Evaluation
A range of regression architectures were evaluated using a chronological 80/20 train/test split (testing on unseen, future time periods).

### Model Performance Summary
Below are the actual validation metrics obtained during training:

| Model Type | Category | RMSE | MAE | $R^2$ Score |
|---|---|---|---|---|
| **Linear Regression** | Statistical | **4.6796** | **2.2829** | **0.8659** |
| **Ridge Regression** | Statistical | 4.6797 | 2.2831 | 0.8659 |
| **XGBoost** | Boosting | 5.8714 | 3.0906 | 0.7891 |
| **Gradient Boosting** | Boosting | 5.9938 | 3.2084 | 0.7802 |
| **Random Forest** | Ensemble | 7.4568 | 3.8614 | 0.6593 |
| **LSTM (Deep Learning)** | Deep Learning | 10.0665 | 7.9150 | 0.3781 |

### Architectural Insights
* **Statistical Baselines**: Linear and Ridge regression models exhibit the best performance (lowest RMSE). This is due to the dominant linear persistence of air quality (`lag_1h_aqi` has a correlation $> 0.95$ with the target variable).
* **LSTM Neural Network**: Implements a recurrent sequence architecture (using past 24 hourly inputs to predict the next hour). The higher RMSE is expected for univariate configurations on highly persistent series, but it provides a robust deep learning representation for non-linear, multi-pollutant patterns.

---

## 6. Model Interpretability (SHAP)
To satisfy explainability guidelines, **SHAP (SHapley Additive exPlanations)** values were computed using the best machine learning model.

### Global Feature Importance
The computed Shapley values reveal the following importances:
1. **`lag_1h_aqi`** (Dominant): Represents immediate persistence. If the air quality is poor right now, it will remain poor in the next hour.
2. **`us_aqi_rolling_24h`**: Captures the daily background trend. High daily baselines indicate slow dispersion rates in valley basins.
3. **`pm2_5_rolling_24h`**: Identifies fine particulate matter ($\text{PM}_{2.5}$) as the key driver of the overall index, highlighting dust and vehicle combustion as primary regional sources.
4. **`hour`**: Reflects the diurnal traffic and heating cycle, capturing traffic peak emissions.

---

## 7. Interactive Web Dashboard
The web application is built on a **FastAPI** backend serving a responsive, modern HTML5 dashboard with vanilla CSS.

### Dashboard Key Features
* **Multi-District Selector**: A sidebar dropdown allows users to switch between any of the 6 Hazara districts, rendering real-time measurements and localized forecasts.
* **Model Leaderboard**: Displays all 6 models, automatically sorted by lowest RMSE, with the active model clearly labeled.
* **Feature Explainability Tab**: Integrates a live, horizontal Plotly bar chart rendering SHAP values, alongside card-based descriptions of feature impacts.
* **Dynamic Health Alerts**: Displays dynamic warning banners for unsafe air conditions (AQI $> 100$) using matching safety colors and advisory warnings.
* **Full Data Export**: Client-side JavaScript polls history and forecasts, compiling them into a unified 72-Hour CSV file for offline research.

---

## 8. MLOps, Automation, & Orchestration
### Bash Controller (`run.sh`)
An easy-to-use controller provides system commands:
* Start server (in background, logging to `server.log`): `./run.sh start`
* Stop server (safely killing the uvicorn socket): `./run.sh stop`
* Restart server: `./run.sh restart`
* Run ML training pipeline: `./run.sh pipeline`
* Verify status: `./run.sh status`

### GitHub Actions Automations
Two CI/CD workflows are configured under `.github/workflows/`:
1. **Hourly Ingestion (`hourly_data_update.yml`)**: Fetches new hourly records from Open-Meteo and appends them to local storage.
2. **Daily Retraining (`daily_retrain.yml`)**: Re-runs training daily, updates model performance tables, regenerates SHAP values, and commits the fresh models back to the registry.

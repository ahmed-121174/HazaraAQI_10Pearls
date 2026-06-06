# Hazara Division AQI Prediction & Monitoring System
## Project Technical Report

---

## 1. Executive Summary
This report details the design, implementation, and deployment of the **Hazara Division Air Quality Index (AQI) Prediction & Monitoring System**. Built for Abbottabad, Haripur, Mansehra, Battagram, Upper Kohistan, and Torghar districts in KPK, Pakistan, this project provides localized, real-time air quality tracking and 72-hour forecasts in mountainous areas.

The project covers the complete machine learning lifecycle:
1. **Data Ingestion**: Automated hourly fetching of weather parameters and air pollutant concentrations from the Open-Meteo API.
2. **Feature Engineering**: Creating historical lag features, rolling statistics, rate of change indicators, and cyclic time variables.
3. **Machine Learning Models**: Training and comparing 6 different models, including linear/ridge regression, tree ensembles (Random Forest, Gradient Boosting, XGBoost), and an LSTM neural network.
4. **Interpretability**: Using SHAP (SHapley Additive exPlanations) to analyze and visualize feature importance.
5. **Web Interface**: A FastAPI dashboard displaying real-time metrics, interactive charts, and options to download full 72-hour history + forecast CSV datasets.
6. **Automation**: A bash script (`run.sh`) to control the application locally, paired with GitHub Actions workflows for automated ingestion and daily retraining.

---

## 2. Geography & Atmospheric Context
Hazara Division spans about 17,197 km² in northern KPK, characterized by deep valleys and high mountain ranges. 

Unlike flat plains or coastal regions, Hazara's air quality is heavily impacted by its geography:
* **Temperature Inversions**: Cold mountain air sinks into the valleys overnight, trapping emissions from traffic and local industries close to the ground (especially in Abbottabad and Haripur).
* **Wind Obstruction**: Steep valleys block strong winds, meaning pollutants do not disperse quickly and remain concentrated for longer periods.
* **Seasonal Factors**: Agricultural burning and winter wood heating lead to seasonal spikes in particulate matter.

### Coordinates Configured
The data pipeline tracks coordinates for the six main districts:
* **Abbottabad (HQ)**: 34.1463° N, 73.2117° E
* **Mansehra**: 34.3302° N, 73.1968° E
* **Haripur**: 33.9942° N, 72.9333° E
* **Battagram**: 34.6837° N, 73.0261° E
* **Upper Kohistan**: 35.2097° N, 73.3456° E
* **Torghar**: 34.6300° N, 72.9400° E

---

## 3. Data Collection & Preprocessing
The ingestion pipeline fetches data from the **Open-Meteo Air Quality API**, tracking:
* Particulate Matter: $\text{PM}_{2.5}$ and $\text{PM}_{10}$
* Gases: Carbon Monoxide ($\text{CO}$), Nitrogen Dioxide ($\text{NO}_2$), Sulphur Dioxide ($\text{SO}_2$), and Ozone ($\text{O}_3$)
* Target variable: US AQI (United States Air Quality Index)

### Data Cleaning
* **Outliers**: Features are clipped at the 1st and 99th percentiles to handle sensor errors or extreme spikes without losing continuous time-series data.
* **Imputation**: Missing values are filled using forward-fill (FFill) to propagate the last active air state, and backward-fill (BFill) as a fallback.

---

## 4. Feature Engineering
To project AQI up to 72 hours forward, we extract 13 key features:
1. **Lags**: `lag_1h_aqi`, `lag_2h_aqi`, and `lag_24h_aqi` capture recent trends and daily cycles.
2. **Rolling Averages**: 6-hour and 24-hour rolling averages (e.g., `us_aqi_rolling_24h`, `pm2_5_rolling_24h`) smooth out minor fluctuations.
3. **Rate of Change**: `aqi_change_rate` calculates the speed of pollution build-up or dispersion.
4. **Time Variables**: `hour`, `day_of_week`, and `month` map diurnal, weekly, and seasonal cycles.

---

## 5. Model Training & Evaluation
We trained 6 models using a chronological 80/20 train/test split to evaluate performance on future, unseen time periods.

### Performance Results
Below are the validation metrics calculated during testing:

| Model Type | Category | RMSE | MAE | $R^2$ Score |
|---|---|---|---|---|
| **Linear Regression** | Statistical | **4.6796** | **2.2829** | **0.8659** |
| **Ridge Regression** | Regularized | 4.6797 | 2.2831 | 0.8659 |
| **XGBoost** | Boosting | 5.8714 | 3.0906 | 0.7891 |
| **Gradient Boosting** | Boosting | 5.9938 | 3.2084 | 0.7802 |
| **Random Forest** | Ensemble | 7.4568 | 3.8614 | 0.6593 |
| **LSTM (Deep Learning)** | Neural Network | 10.0665 | 7.9150 | 0.3781 |

### Insights
* **Linear Models**: Linear and Ridge Regression achieved the lowest RMSE. This is because air quality has strong short-term persistence (`lag_1h_aqi` has a correlation coefficient over 0.95 with the target AQI).
* **LSTM Model**: The LSTM uses the past 24 hours of sequence data to predict the next hour. While its standalone RMSE is higher on this highly persistent dataset, it offers a solid alternative for representing complex, non-linear patterns over time.

---

## 6. Model Interpretability (SHAP)
We used SHAP values to examine how features influence our best-performing regression model:

1. **`lag_1h_aqi`**: This is the most influential feature. Air quality is highly persistent, so current AQI is the strongest predictor of the next hour's AQI.
2. **`us_aqi_rolling_24h`**: Highlights the average background concentration over the last day, accounting for baseline build-up in valley basins.
3. **`pm2_5_rolling_24h`**: Shows that fine particulate matter is the main driver of high AQI readings, linking combustion and dust to local pollution.
4. **`hour`**: Tracks traffic peaks and household heating patterns throughout the day.

---

## 7. Web Dashboard & API
The system features a **FastAPI** web application with a responsive dashboard styled with vanilla CSS.

### Dashboard Features
* **District Selection**: A sidebar menu lets users toggle between the six Hazara districts to view local readings and forecasts.
* **Model Leaderboard**: Lists the metrics for all models, with the top-performing model flagged as active.
* **Explainability Tab**: Integrates an interactive Plotly chart showing SHAP values and feature impacts.
* **Health Advisory Banners**: Color-coded banners automatically display warning levels and health advice based on the current AQI.
* **CSV Exports**: Users can download a full 72-hour dataset combining history and forecasts for offline analysis.

---

## 8. Automation & Orchestration
### Bash CLI Controller (`run.sh`)
Manage the local environment with simple shell commands:
* Start application in background: `./run.sh start`
* Stop application: `./run.sh stop`
* Restart application: `./run.sh restart`
* Run ML training pipeline: `./run.sh pipeline`
* Check status: `./run.sh status`

### GitHub Actions Workflows
Two automated workflows handle background data tasks:
1. **Hourly Ingestion (`hourly_data_update.yml`)**: Fetches the latest hourly data from the API and appends it to the CSV logs.
2. **Daily Retraining (`daily_retrain.yml`)**: Retrains all models once a day, saves the updated model files, and updates performance metrics in the repository.

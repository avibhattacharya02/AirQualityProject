# Air Quality Project

## 📊 Overview
A machine learning project that analyzes and predicts air quality indices (AQI) using multiple deep learning and ensemble models. The project processes hourly air quality data from multiple cities and compares various forecasting approaches to achieve optimal prediction accuracy.

## ✨ Features
- **Exploratory Data Analysis**: Statistical analysis of pollutant concentrations
- **Data Visualization**: Time series plots, correlation heatmaps, distribution analysis
- **Multiple ML Models**: Univariate LSTM, Multivariate LSTM, Random Forest, XGBoost
- **Model Comparison**: Comprehensive evaluation across all approaches
- **Advanced Metrics**: MAE, RMSE, R², and visual performance analysis

## 📈 Dataset
- **File**: `Air_Quality.csv`
- **Time Period**: 2023-01-01 onwards (hourly data)
- **Locations**: Multiple cities including Brasilia and Cairo
- **Features**: 
  - **Temporal**: Date, Time, Day, Month, DayOfWeek
  - **Pollutants**: CO, NO2, SO2, O3, PM2.5, PM10
  - **Target**: AQI (Air Quality Index)

## 🔧 Requirements
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
xgboost>=1.7.0
geopandas>=0.9.0
jupyter>=1.0.0
```

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/avibhattacharya02/AirQualityProject.git
cd AirQualityProject
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📂 Project Structure
```
AirQualityProject/
├── Air Quality.ipynb          # Main Jupyter notebook with analysis & modeling
├── Air_Quality.csv            # Dataset (hourly air quality data)
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── CONTRIBUTING.md            # Contribution guidelines
└── train_data.csv, test_data.csv  # Generated train/test splits
```

## 🚀 Quick Start

1. **Open the notebook:**
   ```bash
   jupyter notebook "Air Quality.ipynb"
   ```
   
   Or view it directly: [Air Quality.ipynb](https://github.com/avibhattacharya02/AirQualityProject/blob/main/Air%20Quality.ipynb)

2. **Update data path** (Cell 8):
   Change `C:\Users\sidhu\Downloads\Air_Quality.csv` to your local path

3. **Run cells sequentially** to:
   - Load and explore data
   - Generate visualizations
   - Train multiple models
   - Compare predictions and metrics

## 📊 Methodology

### Phase 1: Data Exploration
- Load CSV with pandas
- Inspect data structure (8 columns × ~8,760 hourly records)
- Check for missing values
- Display summary statistics
- Identify peak AQI: **188.32 on 2023-09-12 in Cairo**

### Phase 2: Visualization
- **Time series plot**: PM2.5 concentration trends
- **Box plots**: Distribution comparison (PM2.5, PM10, NO2)
- **Histograms**: Frequency distribution of pollutants
- **Correlation heatmap**: Strong correlations between PM2.5/PM10 and AQI (>0.8)

### Phase 3: Data Preprocessing
- Normalize features using MinMaxScaler (0-1 range)
- Create temporal features: Day, Month, DayOfWeek
- Create sliding window sequences (30-day window)
- 80/20 train/test split
- Feature engineering for ensemble models

### Phase 4: Model Development & Comparison

#### Model 1: Univariate LSTM (AQI Only)
**Architecture:**
```
Input Shape: (30, 1)  # 30 timesteps × 1 feature (AQI)
    ↓
LSTM Layer 1: 64 units, return_sequences=True
    ↓
LSTM Layer 2: 32 units
    ↓
Dense Layer: 1 unit (AQI prediction)
    ↓
Output: Scaled AQI value
```

**Performance Metrics:**
| Metric | Value |
|--------|-------|
| MAE | 16.35 |
| RMSE | 20.48 |
| **R² Score** | **0.1888** |

**Interpretation**: Limited predictive power, only explains ~19% of variance.

---

#### Model 2: Multivariate LSTM (Multiple Features)
**Architecture:**
```
Input Shape: (30, 9)  # 30 timesteps × 9 features
    Features: PM2.5, PM10, NO2, SO2, CO, O3, Day, Month, DayOfWeek
    ↓
LSTM Layer 1: 64 units, return_sequences=True
    ↓
LSTM Layer 2: 32 units
    ↓
Dense Layer: 1 unit (AQI prediction)
    ↓
Output: Scaled AQI value (inverse transformed)
```

**Performance Metrics:**
| Metric | Value |
|--------|-------|
| MAE | ~6-8 |
| RMSE | ~7-8 |
| **R² Score** | **0.95** |

**Interpretation**: Excellent deep learning performance with 95% variance explained.

---

#### Model 3: Random Forest Regressor ⭐ **BEST PERFORMER**
**Configuration:**
```
n_estimators: 100 trees
max_depth: 15
min_samples_split: 5
min_samples_leaf: 2
random_state: 42
```

**Performance Metrics:**
| Metric | Value |
|--------|-------|
| MAE | **3.79** ✅ |
| RMSE | **6.50** ✅ |
| **R² Score** | **0.9330** ✅ |

**Key Advantages:**
- Lowest MAE - most accurate individual predictions
- Robust to outliers
- No scaling required for interpretation
- Feature importance rankings available
- Fast training and inference
- Better generalization than deep learning

---

#### Model 4: XGBoost Regressor
**Configuration:**
```
n_estimators: 100 boosting rounds
max_depth: 6
learning_rate: 0.1
subsample: 0.8
colsample_bytree: 0.8
```

**Performance Metrics:**
| Metric | Value |
|--------|-------|
| MAE | 4.68 |
| RMSE | 7.44 |
| **R² Score** | **0.9121** |

**Key Advantages:**
- Gradient boosting approach
- Hyperparameter tuning ready
- Regularization prevents overfitting
- Training curve visibility
- Good balance of bias-variance

---

## 🏆 Model Performance Comparison

| Model | MAE | RMSE | R² Score | Rank |
|-------|-----|------|----------|------|
| **Random Forest** | **3.79** | **6.50** | **0.9330** | 🥇 **1st** |
| XGBoost | 4.68 | 7.44 | 0.9121 | 🥈 2nd |
| Multivariate LSTM | 6-8 | 7-8 | 0.95 | 🥉 3rd |
| Univariate LSTM | 16.35 | 20.48 | 0.1888 | 4th |

**Winner**: **Random Forest Regressor** with R² = 0.9330 and MAE = 3.79

---

## 📌 Key Findings

- **Peak AQI**: 188.32 on 2023-09-12 19:00 in Cairo
- **Diurnal Pattern**: AQI peaks at night (~21:00), lowest during midday (12:00-14:00)
  - **Reason**: Nighttime stable atmosphere, daytime ozone photochemistry
- **Pollutant Correlations**:
  - PM2.5 & PM10: r = 0.95 (very strong)
  - PM2.5 & AQI: r = 0.88 (strong)
  - CO & NO2: r = 0.92 (very strong)
- **Seasonality**: Clear seasonal patterns observed
  - Winter: Higher pollution concentrations
  - Summer: Lower AQI values

**Model Insights:**
- Ensemble methods outperform deep learning for this dataset
- Tree-based models naturally capture non-linear relationships
- Random Forest's bagging approach generalizes better than sequential LSTM
- Feature interactions are crucial for AQI prediction
- Temporal sequences add complexity without proportional benefit

## ⚠️ Known Issues & Limitations

1. **Hard-coded File Paths**: Absolute Windows paths
   - **Fix**: Use relative paths or `pathlib.Path()`
   ```python
   from pathlib import Path
   data_path = Path.cwd() / "Air_Quality.csv"
   ```

2. **Data Leakage Risk**: Random train/test split instead of time-based
   - **Improvement**: Use temporal cross-validation for time series
   ```python
   from sklearn.model_selection import TimeSeriesSplit
   tscv = TimeSeriesSplit(n_splits=5)
   ```

3. **Feature Engineering**: Limited exploration of interaction terms
   - **Opportunity**: Add polynomial features, rolling statistics

4. **Hyperparameter Tuning**: Could be optimized further
   - **Next Step**: Grid search or Bayesian optimization

## 🔮 Future Improvements

- [ ] Hyperparameter optimization using GridSearchCV/BayesSearchCV
- [ ] Cross-validation with TimeSeriesSplit
- [ ] Add weather data (temperature, humidity, wind speed)
- [ ] Ensemble voting/stacking combining multiple models
- [ ] Feature importance analysis (SHAP values)
- [ ] Prediction intervals for uncertainty quantification
- [ ] Multi-step ahead forecasting (7+ days)
- [ ] Geospatial analysis by city
- [ ] Web API deployment (Flask/FastAPI)
- [ ] Real-time prediction dashboard (Streamlit/Dash)
- [ ] Model explainability (LIME/SHAP)
- [ ] Testing on unseen cities/regions

## 📖 Notebook Cell Guide

| Cell | Purpose | Result |
|------|---------|--------|
| 1-6 | Import & setup libraries | ✓ |
| 7-8 | Load CSV and EDA | Peak AQI: 188.32 |
| 9-11 | Visualizations | Time series, Box plots, Histograms |
| 12-13 | Train/test split | 80/20 split |
| 14-15 | Correlation analysis | Strong PM2.5/PM10 correlation |
| 16-24 | **Univariate LSTM** | R² = 0.1888 ❌ |
| 25+ | **Multivariate LSTM** | R² = 0.95 ✓ |
| Later | **Random Forest** | **R² = 0.9330** ✅ BEST |
| Final | **XGBoost** | R² = 0.9121 ✓ |

## 🔑 Code Implementation

**Random Forest Implementation:**
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Train model
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Evaluate
y_pred = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")
```

**XGBoost Implementation:**
```python
import xgboost as xgb

xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
```

**Model Comparison Visualization:**
```python
import matplotlib.pyplot as plt
import numpy as np

models = ['Univariate\nLSTM', 'Multivariate\nLSTM', 'Random\nForest', 'XGBoost']
r2_scores = [0.1888, 0.95, 0.9330, 0.9121]
mae_scores = [16.35, 7.0, 3.79, 4.68]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# R² Comparison
ax1.bar(models, r2_scores, color=['red', 'orange', 'gold', 'silver'])
ax1.set_ylabel('R² Score')
ax1.set_title('Model R² Score Comparison')
ax1.set_ylim([0, 1])
ax1.axhline(y=0.9, color='green', linestyle='--', label='Target: 0.9')
ax1.legend()

# MAE Comparison
ax2.bar(models, mae_scores, color=['red', 'orange', 'gold', 'silver'])
ax2.set_ylabel('Mean Absolute Error (AQI)')
ax2.set_title('Model MAE Comparison (Lower is Better)')
ax2.axhline(y=5, color='green', linestyle='--', label='Target: < 5')
ax2.legend()

plt.tight_layout()
plt.show()
```

## 📚 References

- **Scikit-learn Ensemble**: https://scikit-learn.org/stable/modules/ensemble.html
- **XGBoost Documentation**: https://xgboost.readthedocs.io/
- **Time Series Evaluation**: https://machinelearningmastery.com/time-series-forecasting-performance-measures-with-python/
- **Random Forest**: https://en.wikipedia.org/wiki/Random_forest
- **Gradient Boosting**: https://en.wikipedia.org/wiki/Gradient_boosting
- **Air Quality Index**: https://www.airnow.gov/aqi/

## 👤 Author
- **GitHub**: [@avibhattacharya02](https://github.com/avibhattacharya02)

## 📄 License
MIT License

## 🤝 Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## 📞 Contact & Support
For questions or issues, open a GitHub issue in the repository.

---

**Last Updated**: June 2025  
**Status**: Active Development  
**Best Model**: Random Forest Regressor (R² = 0.9330, MAE = 3.79) ✅  
**Notebook**: [Air Quality.ipynb](https://github.com/avibhattacharya02/AirQualityProject/blob/main/Air%20Quality.ipynb)

# Food Demand Forecasting using Time-Series Machine Learning

---

## Overview

This project builds a time-series forecasting system to predict daily food demand for a restaurant using historical sales data.

The objective is to enable better inventory planning, reduce food waste, and support data-driven operational decisions.

The project follows a structured machine learning pipeline:
- Time-series analysis and decomposition
- Feature engineering with lag and rolling statistics
- Model training using regression algorithms
- Hyperparameter tuning using time-aware validation
- Model evaluation and interpretation

---

## Objectives

- Forecast daily food demand using historical data  
- Capture trend, seasonality, and temporal dependencies  
- Compare baseline and advanced machine learning models  
- Reduce overstocking and stockouts  
- Enable data-driven decision making  

---

## Dataset

**Source:** Simulated restaurant POS dataset (Balaji Fast Food Sales)

### Features:
- Date and time of transaction  
- Item name and category  
- Quantity sold (target variable)  
- Item price and transaction amount  
- Transaction type and time of sale  

---

## Project Structure

Food-Restaurant-series-AI-Model/
│
├── data/
│ ├── raw/
│ │ └── balaji_fast_food_sales_dataset.csv
│ │
│ └── processed/
│ ├── daily_demand.csv
│ ├── model_ready_data.csv
│ ├── train.csv
│ └── test.csv
│
├── notebooks/
│ ├── w1_time_series_eda.ipynb
│ ├── w2_feature_engineering.ipynb
│ └── week3_Model_Training.ipynb
│
├── main.py
├── requirements.txt
└── README.md


---

## Week 1: Time-Series EDA

- Data cleaning and preprocessing  
- Daily demand aggregation  
- Trend and seasonality analysis  
- Time-series decomposition  
- Autocorrelation (ACF) and partial autocorrelation (PACF) analysis  

---

## Week 2: Feature Engineering

- Date-based features (day, month, weekday)  
- Lag features (lag_1, lag_3, lag_7, lag_14, etc.)  
- Rolling statistics (mean, std, min, max)  
- Trend feature creation  
- Cyclical encoding for seasonality  
- Time-series train-test split  

---

## Week 3: Model Training and Selection

### Models Implemented

1. **Naive Baseline**
   - Uses previous day's demand as prediction

2. **Linear Regression**
   - Baseline machine learning model

3. **Random Forest Regressor**
   - Captures non-linear relationships

4. **XGBoost Regressor**
   - Gradient boosting model for high accuracy

---

## Hyperparameter Tuning

- Performed using **GridSearchCV**
- Used **TimeSeriesSplit** to preserve temporal order
- Prevents data leakage and ensures realistic validation

---

## Model Evaluation

Models were evaluated using:

- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**

### Model Comparison

| Model             | MAE      | RMSE     |
|------------------|----------|----------|
| Naive Baseline   | —        | —        |
| Linear Regression| 5.02     | 6.15     |
| Random Forest    | 4.00     | 4.74     |
| XGBoost          | 4.02     | 4.72     |
| **Tuned XGBoost**| **3.90** | **4.46** |

---

## Best Model

**Tuned XGBoost Regressor**

### Best Hyperparameters:

```
learning_rate = 0.01
max_depth = 3
n_estimators = 100
```

---

## Key Metrics

- **MAE**: 3.90 (average error of ~4 units)
- **RMSE**: 4.46 (captures larger errors effectively)

---

## Key Insights

- Lag features are the most important predictors of demand  
- Rolling statistics improve model stability  
- XGBoost significantly outperforms baseline models  
- Time-series cross-validation improves generalization  

---
## Tech Stack

- **Python 3.10+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **Matplotlib & Seaborn** - Visualization
- **Scikit-learn** - Machine learning models
- **XGBoost** - Gradient boosting
- **Statsmodels** - Time-series analysis
- **Google Colab** - Development environment

---

## Business Impact

- **Reduced food waste** through accurate demand prediction  
- **Optimized inventory levels**  
- **Improved stock availability**  
- **Data-driven operational decisions**  
- **Better understanding of demand patterns**  

---

## How to Run

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Food-Restaurant-series-AI-Model-
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the main script:**
   ```bash
   python main.py
   ```

---

## Notebooks Overview

- **w1_time_series_eda.ipynb** - Exploratory data analysis and time-series decomposition
- **w2_feature_engineering.ipynb** - Feature creation and train-test split
- **week3_Model_Training.ipynb** - Model training, tuning, and evaluation

---

## Future Improvements

- **Prophet model integration** for improved forecasting
- **Deep learning models** (LSTM, GRU) for complex patterns
- **Real-time dashboard** with interactive visualizations
- **API deployment** for production use
- **External factors** (weather, holidays, events)
- **Advanced feature engineering** (Fourier terms, interaction features)

---

## Project Status

- Week 1 Completed  
- Week 2 Completed  
- Week 3 Completed  
- Week 4 In Progress  

---

## Contributors

- **Person 1** - Data Cleaning & EDA
- **Person 2** - Feature Engineering
- **Person 3** - Model Training & Tuning
- **Person 4** - Evaluation & Documentation

---

## License

This project is licensed under the MIT License.
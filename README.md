# 🍔 Food Demand Forecasting & Inventory Optimization (AI Project)

# 📌 Overview

This project focuses on building an AI-powered demand forecasting system for a restaurant/food service business. The goal is to predict future demand using historical POS (Point of Sale) data to reduce food waste and optimize inventory.

---

# 🎯 Objectives

* Forecast daily food demand using time-series modeling
* Identify trends, seasonality, and demand patterns
* Reduce overstocking and stockouts
* Support data-driven decision making

---

# 📊 Dataset

**Source:** Simulated restaurant POS data (Balaji Fast Food Sales)

**Features include:**

* Date & Time
* Item Name & Category
* Quantity Sold
* Item Price
* Transaction Amount
* Transaction Type
* Time of Sale
* Customer Information

---

# 🏗️ Project Structure

```
food-demand-forecasting/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── EDA_Notebook_Balaji.ipynb
│   ├── Person3_Model_Training.ipynb
│
├── src/
├── models/
├── reports/
│
├── requirements.txt
├── README.md
└── main.py
```

---

# ✅ Week 1 Progress (Completed)

* Data cleaning and preprocessing
* Time-series dataset creation
* Trend analysis
* Seasonality detection
* Time-series decomposition
* Autocorrelation analysis (ACF, PACF)

---

# ✅ Week 2 Progress (Completed)

* Feature Engineering
* Date-based features
* Lag features creation
* Rolling mean features
* Model-ready dataset preparation
* Train-Test split (Time-series based)

---

# ✅ Week 3 Progress (Completed — Person 3)

## 🤖 Model Training & Tuning

Three regression models were implemented:

### 1. Linear Regression

Baseline model for demand forecasting

### 2. Random Forest Regressor

Handled non-linear demand patterns

### 3. XGBoost Regressor

Advanced gradient boosting model for improved accuracy

---

# 📊 Model Evaluation

Models evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

## Model Comparison

| Model             | MAE      | RMSE     |
| ----------------- | -------- | -------- |
| Linear Regression | 5.02     | 6.15     |
| Random Forest     | 4.00     | 4.74     |
| XGBoost           | 4.02     | 4.72     |
| **Tuned XGBoost** | **3.90** | **4.46** |

---

# 🏆 Best Model Selected

**Tuned XGBoost Regressor**

Best Hyperparameters:

```
learning_rate = 0.01
max_depth = 3
n_estimators = 100
```

---

# 📈 Key Metrics

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

---

# 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Statsmodels
* Google Colab

---

# 🚀 Business Impact

* Reduced food waste
* Improved inventory planning
* Better demand prediction
* Data-driven restaurant operations

---

# ▶️ How to Run

Install dependencies:

```
pip install -r requirements.txt
```

Run project:

```
python main.py
```

---

# 👨‍💻 Contributors

* Person 1 — Data Cleaning & EDA
* Person 2 — Feature Engineering
* Person 3 — Model Training & Tuning
* Person 4 — Evaluation & Documentation

---

# 📌 Future Improvements

* Prophet Model Integration
* Deep Learning (LSTM) Forecasting
* Real-time Dashboard
* API Deployment

---

# ⭐ Project Status

✅ Week 1 Completed
✅ Week 2 Completed
✅ Week 3 Completed
⬜ Week 4 In Progress

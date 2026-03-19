# 🍔 Food Demand Forecasting & Inventory Optimization (AI Project)

## 📌 Overview

This project focuses on building an AI-powered demand forecasting system for a restaurant/food service business.
The goal is to predict future demand using historical POS (Point of Sale) data to reduce food waste and optimize inventory.

---

## 🎯 Objectives

* Forecast daily food demand using time-series modeling
* Identify trends, seasonality, and demand patterns
* Reduce overstocking and stockouts
* Support data-driven decision making

---

## 📊 Dataset

* Source: Simulated restaurant POS data
* Size: ~20,000 records
* Features include:

  * Date, Time
  * Item Name, Category
  * Quantity Sold
  * Price, Profit
  * Customer & Order Details

---

## 🏗️ Project Structure

```
food-demand-forecasting/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│
├── models/
│
├── reports/
│
├── requirements.txt
├── README.md
└── main.py
```

---

## ✅ Week 1 Progress (Completed)

* Data cleaning and preprocessing
* Time-series dataset creation (daily demand)
* Trend and seasonality analysis
* Time-series decomposition
* Autocorrelation analysis (ACF, PACF)

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Statsmodels
* Scikit-learn

---

## 📈 Key Metrics

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python main.py
```

---

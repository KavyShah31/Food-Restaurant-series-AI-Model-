import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


# ---------------------------
# LOAD DATA (ROBUST PATH)
# ---------------------------
@st.cache_data
def load_data():
    # Go from src/ → project root
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    file_path = os.path.join(base_path, "data", "processed", "model_ready_data.csv")

    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        st.stop()

    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])

    return df


# ---------------------------
# TRAIN MODEL (CACHED)
# ---------------------------
@st.cache_resource
def train_model(X_train, y_train, n_estimators, max_depth, learning_rate):
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------
# MAIN APP
# ---------------------------
def main():
    st.set_page_config(page_title="Food Demand Forecasting", layout="wide")

    st.title("Food Demand Forecasting Dashboard")

    # ---------------------------
    # LOAD DATA
    # ---------------------------
    df = load_data()

    # ---------------------------
    # SIDEBAR CONTROLS
    # ---------------------------
    st.sidebar.header("Controls")

    # Date filter
    start_date = st.sidebar.date_input("Start Date", df['date'].min())
    end_date = st.sidebar.date_input("End Date", df['date'].max())

    df = df[(df['date'] >= pd.to_datetime(start_date)) &
            (df['date'] <= pd.to_datetime(end_date))]

    # Train split slider
    split_ratio = st.sidebar.slider("Train Size (%)", 60, 90, 80)

    # Model parameters
    n_estimators = st.sidebar.slider("n_estimators", 50, 300, 100)
    max_depth = st.sidebar.slider("max_depth", 2, 10, 3)
    learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.3, 0.01)

    # ---------------------------
    # DATA OVERVIEW
    # ---------------------------
    st.subheader("Dataset Overview")
    st.write("Shape:", df.shape)
    st.write("Date Range:", df['date'].min(), "to", df['date'].max())

    # ---------------------------
    # PREPARE DATA
    # ---------------------------
    y = df['quantity']
    X = df.drop(['quantity', 'date'], axis=1)

    train_size = int(len(X) * (split_ratio / 100))

    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:]

    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    # ---------------------------
    # TRAIN MODEL
    # ---------------------------
    model = train_model(X_train, y_train, n_estimators, max_depth, learning_rate)
    pred = model.predict(X_test)

    # ---------------------------
    # BASELINE (NAIVE)
    # ---------------------------
    naive_pred = X_test['lag_1']

    mae_naive = mean_absolute_error(y_test, naive_pred)
    rmse_naive = np.sqrt(mean_squared_error(y_test, naive_pred))

    # ---------------------------
    # MODEL METRICS
    # ---------------------------
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    st.subheader("Model Performance")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model MAE", round(mae, 2))
    col2.metric("Model RMSE", round(rmse, 2))
    col3.metric("Naive MAE", round(mae_naive, 2))
    col4.metric("Naive RMSE", round(rmse_naive, 2))

    # ---------------------------
    # ACTUAL VS PREDICTED
    # ---------------------------
    st.subheader("Actual vs Predicted Demand")

    fig, ax = plt.subplots()
    ax.plot(y_test.reset_index(drop=True), label="Actual")
    ax.plot(pd.Series(pred), label="Predicted")
    ax.legend()

    st.pyplot(fig)

    # ---------------------------
    # ERROR ANALYSIS
    # ---------------------------
    st.subheader("Prediction Error Distribution")

    errors = y_test.values - pred

    fig_err, ax_err = plt.subplots()
    ax_err.hist(errors, bins=30)
    ax_err.set_title("Error Distribution")

    st.pyplot(fig_err)

    # ---------------------------
    # FEATURE IMPORTANCE
    # ---------------------------
    st.subheader("Feature Importance")

    importance = model.feature_importances_
    features = X.columns

    feat_imp = pd.Series(importance, index=features).sort_values()

    fig2, ax2 = plt.subplots()
    feat_imp.plot(kind='barh', ax=ax2)

    st.pyplot(fig2)

    # ---------------------------
    # SAMPLE PREDICTIONS
    # ---------------------------
    st.subheader("Sample Predictions")

    results_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": pred,
        "Error": errors
    })

    st.dataframe(results_df.head(20))


# ---------------------------
# RUN APP
# ---------------------------
if __name__ == "__main__":
    main()
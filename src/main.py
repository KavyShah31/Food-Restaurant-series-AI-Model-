import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="Food Demand Forecasting", layout="wide")


# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_path, "data", "processed", "model_ready_data.csv")

    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        st.stop()

    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])

    return df


# ---------------------------
# LOAD MODEL
# ---------------------------
@st.cache_resource
def load_model():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_path, "models", "xgboost_model.pkl")

    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


# ---------------------------
# TRAIN MODEL (FALLBACK)
# ---------------------------
@st.cache_resource
def train_model(X_train, y_train):
    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.01,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------
# MAIN APP
# ---------------------------
def main():
    st.title("Food Demand Forecasting Dashboard")

    # ---------------------------
    # LOAD DATA
    # ---------------------------
    df = load_data()

    # ---------------------------
    # SIDEBAR
    # ---------------------------
    st.sidebar.header("Controls")

    start_date = st.sidebar.date_input("Start Date", df['date'].min())
    end_date = st.sidebar.date_input("End Date", df['date'].max())

    df = df[(df['date'] >= pd.to_datetime(start_date)) &
            (df['date'] <= pd.to_datetime(end_date))]

    split_ratio = st.sidebar.slider("Train Size (%)", 60, 90, 80)

    # ---------------------------
    # MODEL INFO PANEL (NEW)
    # ---------------------------
    st.sidebar.markdown("### Model Information")
    st.sidebar.info(
        """
        Model: Tuned XGBoost  
        Validation: TimeSeriesSplit  
        Features: Lag + Rolling Statistics  
        Target: Daily Quantity Sold  
        """
    )

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
    # LOAD / TRAIN MODEL
    # ---------------------------
    model = load_model()

    if model is not None:
        st.success("Using pre-trained model")
    else:
        st.warning("Pre-trained model not found. Training model...")
        model = train_model(X_train, y_train)

    pred = model.predict(X_test)

    # ---------------------------
    # BASELINE
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
    # ERROR DISTRIBUTION
    # ---------------------------
    st.subheader("Prediction Error Distribution")

    errors = y_test.values - pred

    fig_err, ax_err = plt.subplots()
    ax_err.hist(errors, bins=30)

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
# RUN
# ---------------------------
if __name__ == "__main__":
    main()
import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

st.set_page_config(page_title="Food Demand Forecasting", layout="wide")


# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_path, "data", "processed", "model_ready_data.csv")

    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    return df


# ---------------------------
# LOAD MODEL
# ---------------------------
@st.cache_resource
def load_model():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_path, "models", "xgboost_model.pkl")

    if os.path.exists(model_path):
        saved = joblib.load(model_path)
        if isinstance(saved, dict):
            return saved["model"], saved["features"]
        return saved, None

    return None, None


# ---------------------------
# TRAIN MODEL (Fallback)
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
# FORECAST FUNCTION (NEW 🔮)
# ---------------------------
def forecast_future(model, df, features, days=7):
    df = df.copy()
    future_preds = []

    for _ in range(days):
        last_row = df.iloc[-1:].copy()

        pred = model.predict(last_row[features])[0]
        future_preds.append(pred)

        new_row = last_row.copy()
        new_row['quantity'] = pred

        # Update lag features dynamically
        for lag in [1, 3, 7, 14]:
            col = f'lag_{lag}'
            if col in df.columns and len(df) >= lag:
                new_row[col] = df['quantity'].iloc[-lag]

        df = pd.concat([df, new_row], ignore_index=True)

    return future_preds


# ---------------------------
# MAIN APP
# ---------------------------
def main():
    st.title("📊 Food Demand Forecasting Dashboard")

    df = load_data()

    # ---------------------------
    # SIDEBAR
    # ---------------------------
    st.sidebar.header("⚙️ Controls")

    mode = st.sidebar.radio(
        "Select Mode",
        ["📈 Model Evaluation", "🔮 Forecast Future Demand"]
    )

    # ---------------------------
    # DATA PREP
    # ---------------------------
    y = df['quantity']
    X = df.drop(['quantity', 'date'], axis=1)

    split_ratio = 80
    train_size = int(len(X) * (split_ratio / 100))

    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:]

    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    # ---------------------------
    # MODEL
    # ---------------------------
    model, model_features = load_model()

    if model is None:
        st.warning("No saved model found. Training new model...")
        model = train_model(X_train, y_train)
        model_features = X_train.columns.tolist()

    if model_features:
        X_test = X_test[model_features]

    pred = model.predict(X_test)

    # ===========================
    # 📈 MODE 1: MODEL EVALUATION
    # ===========================
    if mode == "📈 Model Evaluation":

        st.subheader("📊 Model Performance")

        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))

        naive_pred = X_test['lag_1']
        mae_naive = mean_absolute_error(y_test, naive_pred)

        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", round(mae, 2))
        col2.metric("RMSE", round(rmse, 2))
        col3.metric("Baseline MAE", round(mae_naive, 2))

        if mae < mae_naive:
            st.success("Model outperforms baseline → Reliable")
        else:
            st.error("Model underperforms baseline")

        # Plot
        st.subheader("📉 Actual vs Predicted")

        fig, ax = plt.subplots()
        ax.plot(y_test.values, label="Actual")
        ax.plot(pred, label="Predicted", linestyle='--')
        ax.legend()
        st.pyplot(fig)

        # Feature Importance
        st.subheader("🔍 Feature Importance")

        if hasattr(model, "feature_importances_"):
            feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()

            fig2, ax2 = plt.subplots()
            feat_imp.plot(kind='barh', ax=ax2)
            st.pyplot(fig2)

            st.info(f"Top driver: {feat_imp.index[-1]}")

    # ===========================
    # 🔮 MODE 2: FORECASTING
    # ===========================
    else:

        st.subheader("🔮 Future Demand Forecast")

        forecast_days = st.slider("Forecast Horizon (days)", 1, 14, 7)

        if st.button("Generate Forecast"):

            preds = forecast_future(model, df, X.columns.tolist(), forecast_days)

            st.success(f"Forecast generated for next {forecast_days} days")

            # Show values
            forecast_df = pd.DataFrame({
                "Day": range(1, forecast_days + 1),
                "Predicted Demand": preds
            })

            st.dataframe(forecast_df)

            # Plot
            st.subheader("📈 Forecast Visualization")

            fig, ax = plt.subplots()

            recent_actual = y.tail(30).values

            ax.plot(range(len(recent_actual)), recent_actual, label="Recent Actual")

            ax.plot(
                range(len(recent_actual), len(recent_actual) + forecast_days),
                preds,
                label="Forecast",
                linestyle='--'
            )

            ax.legend()
            st.pyplot(fig)

            # Business Insight
            st.subheader("💼 Business Insight")

            avg_demand = np.mean(preds)

            st.info(f"""
            Average predicted demand for next {forecast_days} days: {round(avg_demand, 2)}

            👉 Use this to:
            - Plan inventory accordingly
            - Reduce food waste
            - Optimize staffing
            """)


# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="SaharMart CLV Predictor", layout="wide")

st.title("📊 SaharMart CLV Prediction App")

# Upload data
uploaded_file = st.file_uploader("Upload your SaharMart CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Dataset")
    st.dataframe(df.head())

    # Drop IDs
    df_model = df.drop(columns=["CustomerID", "CustomerName"])

    # One-hot encode
    df_model = pd.get_dummies(df_model, drop_first=True)

    # Split features and target
    X = df_model.drop("CustomerLifetimeValue", axis=1)
    y = df_model["CustomerLifetimeValue"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Ridge model
    ridge = Ridge()
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)

    # Metrics
    st.subheader("📈 Model Evaluation Metrics")
    st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred):.2f}")
    st.write(f"**RMSE:** {mean_squared_error(y_test, y_pred, squared=False):.2f}")
    st.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")

    # Plot: Actual vs Predicted
    st.subheader("🔍 Actual vs Predicted CLV")
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual CLV")
    ax.set_ylabel("Predicted CLV")
    st.pyplot(fig)

    # Plot: Residuals
    st.subheader("📉 Prediction Error Distribution")
    residuals = y_test - y_pred
    fig2, ax2 = plt.subplots()
    sns.histplot(residuals, kde=True, ax=ax2)
    ax2.set_title("Residual Distribution")
    st.pyplot(fig2)


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# App configuration
st.set_page_config(page_title="SaharMart CLV Predictor", layout="wide")

st.title("📊 SaharMart Customer Lifetime Value (CLV) Predictor")

# Upload Section
st.markdown("### 📤 Upload Your Dataset")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Preview the data
        st.subheader("🔍 Data Preview")
        st.dataframe(df.head())

        st.markdown("----")
        st.subheader("📊 Exploratory Data Analysis")

        # Show data types
        st.markdown("**Data Types:**")
        st.write(df.dtypes)

        # Correlation heatmap for numeric features
        st.markdown("**Correlation Heatmap:**")
        df_corr = df.select_dtypes(include=['int64', 'float64'])
        fig_corr, ax_corr = plt.subplots()
        sns.heatmap(df_corr.corr(), annot=True, cmap="coolwarm", ax=ax_corr)
        st.pyplot(fig_corr)

        # Modeling Section
        st.markdown("----")
        st.subheader("⚙️ Ridge Regression Modeling")

        # Drop non-useful columns
        df_model = df.drop(columns=["CustomerID", "CustomerName"], errors="ignore")

        # Encode categorical features
        df_model = pd.get_dummies(df_model, drop_first=True)

        # Split features and target
        if "CustomerLifetimeValue" in df_model.columns:
            X = df_model.drop("CustomerLifetimeValue", axis=1)
            y = df_model["CustomerLifetimeValue"]

            # Scale numerical features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Split into train/test sets
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            # Ridge model
            model = Ridge()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Show model metrics
            st.subheader("📈 Model Evaluation Metrics")
            st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred):.2f}")
            st.write(f"**RMSE:** {mean_squared_error(y_test, y_pred, squared=False):.2f}")
            st.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")

            # Actual vs predicted plot
            st.subheader("🔍 Actual vs Predicted CLV")
            fig_pred, ax_pred = plt.subplots()
            sns.scatterplot(x=y_test, y=y_pred, ax=ax_pred)
            ax_pred.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax_pred.set_xlabel("Actual CLV")
            ax_pred.set_ylabel("Predicted CLV")
            st.pyplot(fig_pred)

            # Residuals plot
            st.subheader("📉 Residuals (Prediction Error Distribution)")
            residuals = y_test - y_pred
            fig_resid, ax_resid = plt.subplots()
            sns.histplot(residuals, kde=True, ax=ax_resid)
            st.pyplot(fig_resid)
        else:
            st.error("Target variable 'CustomerLifetimeValue' not found in dataset.")

    except Exception as e:
        st.error(f"Something went wrong while processing your file: {e}")
else:
    st.info("Please upload a CSV file to begin.")

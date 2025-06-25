
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="SaharMart CLV Predictor", layout="wide")

st.title("📊 SaharMart Customer Lifetime Value Predictor")

uploaded_file = st.file_uploader("Upload your SaharMart CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Raw Data")
    st.write(data.head())

    with st.expander("📈 Data Exploration"):
        st.write("Data Summary:")
        st.write(data.describe(include='all'))
        st.write("Data Info:")
        st.write(data.dtypes)
        st.write("Duplicated rows:", data.duplicated().sum())

        fig1, ax1 = plt.subplots()
        sns.histplot(data['CustomerLifetimeValue'], kde=True, ax=ax1)
        st.pyplot(fig1)

        st.write("Customer Segment Count:")
        st.bar_chart(data['CustomerSegment'].value_counts())

        fig2, ax2 = plt.subplots()
        sns.boxplot(x='CustomerSegment', y='CustomerLifetimeValue', data=data, ax=ax2)
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots()
        sns.scatterplot(x='TenureMonths', y='CustomerLifetimeValue', data=data, ax=ax3)
        st.pyplot(fig3)

        numerical_data = data.drop(['CustomerID', 'CustomerName', 'Region', 'ReferralSource', 'CustomerSegment'], axis=1)
        fig4, ax4 = plt.subplots()
        sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm', ax=ax4)
        st.pyplot(fig4)

    with st.expander("⚙️ Model Training"):
        data = data.drop(columns=['CustomerID', 'CustomerName'])
        data = pd.get_dummies(data, columns=['Region', 'ReferralSource', 'CustomerSegment'], drop_first=True)

        X = data.drop(columns=['CustomerLifetimeValue'])
        y = data['CustomerLifetimeValue']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.write("Model Evaluation Metrics:")
        st.write("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
        st.write("R² Score:", r2_score(y_test, y_pred))

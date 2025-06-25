# 🛍️ SaharMart Customer Lifetime Value (CLV) Prediction

This project builds a **supervised machine learning model** to predict **Customer Lifetime Value (CLV)** for a fictional e-commerce business called **SaharMart**. The goal is to help business teams understand which customers are most valuable over time — and retain them better through smarter decisions.

---

## 🚀 Project Overview

📌 **Problem:**  
SaharMart wants to predict how much revenue a customer is likely to generate in their lifetime, so they can tailor marketing, rewards, and retention strategies.

📌 **Solution:**  
We built a regression model (Ridge Regression) to predict CLV using features like:
- Region
- Tenure with the company
- Average monthly spend
- Purchase frequency
- Referral source
- Returns
- Customer segment

---

## 🧠 Machine Learning Pipeline

✅ **1. Data Loading & Cleaning**  
✅ **2. Exploratory Data Analysis (EDA)**  
✅ **3. Feature Engineering & Encoding**  
✅ **4. Scaling with StandardScaler**  
✅ **5. Ridge Regression Modeling**  
✅ **6. Model Evaluation** using MAE, RMSE, and R²  
✅ **7. Visualizations**: Correlation heatmap, Prediction vs Actual, Residuals

---

## 🛠️ Tools & Libraries Used

| Tool           | Purpose                         |
|----------------|----------------------------------|
| `Python`       | Core programming language        |
| `Pandas`       | Data manipulation                |
| `NumPy`        | Numeric operations               |
| `Matplotlib`   | Visualization                    |
| `Seaborn`      | Advanced visualization           |
| `Scikit-learn` | ML modeling and evaluation       |
| `Streamlit`    | Web app to demo the model        |

---

## 📊 Streamlit App Demo

We built a user-friendly app with **Streamlit** where users can:

- Upload their own dataset
- Preview and analyze the data
- See predicted vs actual CLV
- View model performance metrics

💡 **To run it locally:**

```bash
pip install -r requirements.txt
streamlit run saharmart_clv_app.py

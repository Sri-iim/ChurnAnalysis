import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Set up Streamlit UI
st.set_page_config(page_title="📊 Customer Churn Prediction", layout="wide")

# Load dataset from local path
file_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"  # Update if needed
df = pd.read_csv(file_path)

# Function to preprocess data
def preprocess_data(df):
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.fillna(0, inplace=True)

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
        if df[col].nunique() < 10:
            df[col] = df[col].astype("category").cat.codes

    return df

# Preprocess dataset
df = preprocess_data(df)

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
option = st.sidebar.radio("Go to:", ["🏠 Home", "📊 Data Overview", "📈 Visualizations", "🤖 Model & Prediction"])

# Home Page
if option == "🏠 Home":
    st.title("📊 Customer Churn Prediction App")
    st.write("Welcome to the interactive app for analyzing and predicting customer churn.")
    st.image("https://miro.medium.com/max/1400/1*QXt5jkpXbaXhxeNzWVeF6w.png", use_column_width=True)
    st.write("🔹 Explore customer data, analyze churn trends, and make predictions.")

# Data Overview Page
elif option == "📊 Data Overview":
    st.title("📊 Dataset Overview")
    st.write("### Sample Data")
    st.dataframe(df.head())

    st.write("### Data Summary")
    st.write(df.describe())

    st.write("### Missing Values")
    missing_values = df.isnull().sum()
    st.write(missing_values[missing_values > 0])

# Visualizations Page
elif option == "📈 Visualizations":
    st.title("📊 Data Visualizations")

    # Churn Distribution
    st.write("### Churn Distribution")
    fig, ax = plt.subplots()
    df["Churn"].value_counts().plot.pie(autopct="%1.1f%%", startangle=90, colors=["green", "red"], ax=ax)
    st.pyplot(fig)

    # Monthly Charges Distribution
    st.write("### Monthly Charges Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["MonthlyCharges"], bins=30, kde=True, color="blue", ax=ax)
    st.pyplot(fig)

    # Correlation Heatmap
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

# Model & Prediction Page
elif option == "🤖 Model & Prediction":
    st.title("🤖 Train Model & Predict Churn")

    # Train-Test Split
    X = df.drop(columns=["Churn"], errors="ignore")
    y = df["Churn"].astype("category").cat.codes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "churn_model.pkl")

    # Model Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.write(f"✅ **Model Accuracy:** {accuracy:.2%}")

    # Confusion Matrix
    st.write("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    # User Input for Prediction
    st.subheader("🔍 Predict Customer Churn")
    user_input = {}
    for col in X.columns:
        user_input[col] = st.number_input(f"{col}", value=float(X[col].median()))

    # Convert input to DataFrame
    user_df = pd.DataFrame([user_input])

    # Make prediction
    if st.button("Predict"):
        prediction = model.predict(user_df)[0]
        result = "Churn" if prediction == 1 else "No Churn"
        st.success(f"**Prediction: {result}**")

st.sidebar.write("📌 Made with ❤️ by [Your Name]")

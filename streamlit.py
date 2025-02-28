import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

@st.cache_data
def load_data():
    df = pd.read_csv("customer_churn.csv")
    return df

def preprocess_data(df):
    df = df.drop(columns=['customerID'], errors='ignore')  # Drop if exists

    # Encode categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    label_encoders = {col: LabelEncoder().fit(df[col]) for col in categorical_cols}
    for col, le in label_encoders.items():
        df[col] = le.transform(df[col])

    # Separate features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Scale numerical features
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    return X, y, label_encoders, scaler

@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc

def user_input_features(label_encoders, scaler):
    st.sidebar.header("Enter Customer Details")

    # User input fields
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 24)
    MonthlyCharges = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)
    TotalCharges = st.sidebar.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1000.0)

    # Categorical inputs
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
    Partner = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
    Dependents = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
    PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    Contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    PaymentMethod = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    # Create dataframe
    input_data = pd.DataFrame([[tenure, MonthlyCharges, TotalCharges, gender, SeniorCitizen, Partner, Dependents,
                                PhoneService, InternetService, Contract, PaymentMethod]],
                              columns=["tenure", "MonthlyCharges", "TotalCharges", "gender", "SeniorCitizen",
                                       "Partner", "Dependents", "PhoneService", "InternetService", "Contract", "PaymentMethod"])

    # Apply label encoding
    for col in label_encoders:
        input_data[col] = label_encoders[col].transform(input_data[col])

    # Scale numerical columns
    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

    return input_data

# Load data
df = load_data()

# Preprocess data
X, y, label_encoders, scaler = preprocess_data(df)

# Train model
model, accuracy = train_model(X, y)

# Get user input
input_data = user_input_features(label_encoders, scaler)

# Predict churn
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

# Display results
st.write(f"### Model Accuracy: {accuracy:.2%}")
st.write("### Prediction Results")
st.write(f"Churn Prediction: {'Yes' if prediction[0] == 1 else 'No'}")
st.write(f"Churn Probability: {prediction_proba[0][1]:.2%}")

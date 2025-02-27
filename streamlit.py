import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

# Load dataset (Replace with your dataset path)
df = pd.read_csv("your_dataset.csv")

# Ensure dataset is cleaned and processed correctly
df.dropna(inplace=True)

# Splitting dataset into features and target
X = df.drop(columns=['Churn'])  # Feature columns
y = df['Churn']  # Target column

# Encoding categorical features
X_encoded = pd.get_dummies(X, drop_first=True)

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# Standardizing numerical columns
scaler = StandardScaler()
numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AverageMonthlySpend', 'TotalServices']

# Apply transformation and explicitly assign back to DataFrame
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns].values)
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns].values)

# Streamlit App UI
st.title("Customer Churn Prediction App")
st.write("Predict whether a customer will churn based on various factors.")

# Model Training Function
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    st.write(f"### {model_name} Evaluation")
    st.write(f"**Accuracy:** {accuracy:.4f}")
    st.write(f"**Precision:** {precision:.4f}")
    st.write(f"**Recall:** {recall:.4f}")
    st.write(f"**F1-Score:** {f1:.4f}")
    st.write(f"**ROC-AUC:** {roc_auc:.4f}")
    st.write("**Confusion Matrix:**")
    st.write(confusion_matrix(y_test, y_pred))
    st.write("**Classification Report:**")
    st.write(classification_report(y_test, y_pred))
    
    return model

# Training Models
log_reg = LogisticRegression(random_state=42)
rf = RandomForestClassifier(random_state=42)

st.subheader("Logistic Regression Evaluation")
log_reg = evaluate_model(log_reg, X_train, X_test, y_train, y_test, "Logistic Regression")

st.subheader("Random Forest Evaluation")
rf = evaluate_model(rf, X_train, X_test, y_train, y_test, "Random Forest")

# User Input for Prediction
st.subheader("Churn Prediction")

# Creating input fields for user
tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=50.0)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=600.0)
avg_monthly_spend = st.number_input("Average Monthly Spend ($)", min_value=0.0, max_value=500.0, value=60.0)
total_services = st.number_input("Total Services Used", min_value=1, max_value=10, value=3)

# Convert input into numpy array
input_data = np.array([[tenure, monthly_charges, total_charges, avg_monthly_spend, total_services]])

# Predict button
if st.button("Predict Churn"):
    input_data = scaler.transform(input_data)  # Standardize input

    prediction_log_reg = log_reg.predict(input_data)
    prediction_rf = rf.predict(input_data)

    st.write(f"Logistic Regression Prediction: {'Churn' if prediction_log_reg[0] == 1 else 'No Churn'}")
    st.write(f"Random Forest Prediction: {'Churn' if prediction_rf[0] == 1 else 'No Churn'}")

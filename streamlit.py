import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# Load the dataset
@st.cache_data
def load_data():
    url = "WA_Fn-UseC_-Telco-Customer-Churn.csv"  # Replace with your actual path or URL
    df = pd.read_csv(url)
    return df

df = load_data()

# Print columns for debugging
st.write("Initial DataFrame columns:", df.columns.tolist())

# Data Preprocessing and Feature Engineering
def preprocess_data(df):
    # Ensure 'customerID' is dropped if it exists
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)
    
    # Convert 'TotalCharges' to numeric, handling errors
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Convert 'tenure' to integer
    df['tenure'] = df['tenure'].astype(int)
    
    # Define categorical columns
    categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                           'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                           'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                           'PaperlessBilling', 'PaymentMethod', 'Churn']
    
    # Check for missing columns and handle them
    missing_cols = [col for col in categorical_columns if col not in df.columns]
    if missing_cols:
        st.warning(f"Missing columns: {missing_cols}. These columns will be skipped.")
        categorical_columns = [col for col in categorical_columns if col not in missing_cols]
    
    # Convert categorical columns to 'category' dtype
    if categorical_columns:
        df[categorical_columns] = df[categorical_columns].astype('category')
    
    # Fill missing 'TotalCharges' with median
    total_charges_median = df['TotalCharges'].median()
    df['TotalCharges'] = df['TotalCharges'].fillna(total_charges_median)
    
    # Fill missing values in categorical columns with mode
    for col in categorical_columns:
        if df[col].isnull().any():
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value)
    
    # Create a new feature for phone service and multiple lines
    df['PhoneService_MultipleLines'] = df.apply(
        lambda row: 'No Phone Service' if row['PhoneService'] == 'No' else row['MultipleLines'], axis=1
    )
    
    # Calculate total services
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['TotalServices'] = df[services].apply(lambda row: (row == 'Yes').sum(), axis=1)

    # Create tenure category
    def tenure_category(tenure):
        if tenure <= 12:
            return '0-12 months'
        elif tenure <= 24:
            return '12-24 months'
        elif tenure <= 36:
            return '24-36 months'
        elif tenure <= 48:
            return '36-48 months'
        elif tenure <= 60:
            return '48-60 months'
        else:
            return '60+ months'

    df['TenureCategory'] = df['tenure'].apply(tenure_category)
    
    # Calculate average monthly spend
    df['AverageMonthlySpend'] = df.apply(
        lambda row: row['TotalCharges'] / row['tenure'] if row['tenure'] > 0 else 0, axis=1
    )
    
    # Drop unnecessary columns
    df.drop(columns=['PhoneService', 'MultipleLines'] + services, inplace=True, errors='ignore')
    
    # Map 'Yes'/'No' to 1/0 for binary columns
    yes_no_columns = ['Partner', 'Dependents', 'PaperlessBilling', 'Churn']
    for col in yes_no_columns:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna (mode_value).astype(int)
    
    # Fill missing values in numerical columns with median
    numerical_columns = ['MonthlyCharges', 'TotalCharges', 'AverageMonthlySpend']
    for col in numerical_columns:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)

    return df

# Preprocess the data
df = preprocess_data(df)

# Display the processed DataFrame
st.write("Processed DataFrame:", df.head())

# Proceed with model training or further analysis
X = df.drop(columns=['Churn'])
y = df['Churn']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = logistic_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Display evaluation metrics
st.write("Model Evaluation Metrics:")
st.write(f"Accuracy: {accuracy:.2f}")
st.write(f"Precision: {precision:.2f}")
st.write(f"Recall: {recall:.2f}")
st.write(f"F1 Score: {f1:.2f}")
st.write(f"ROC AUC: {roc_auc:.2f}")

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
st.write("Confusion Matrix:")
st.write(conf_matrix)

# Display classification report
class_report = classification_report(y_test, y_pred)
st.text("Classification Report:")
st.text(class_report)

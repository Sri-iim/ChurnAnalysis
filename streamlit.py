import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    return df

# Preprocess data
def preprocess_data(df):
    df.drop(columns=['customerID'], inplace=True)
    df.dropna(inplace=True)
    
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() == 2:
            df[col] = LabelEncoder().fit_transform(df[col])
    
    df = pd.get_dummies(df, drop_first=True)
    return df

# Train churn prediction model
def train_model(df):
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    return model, scaler, acc

# Streamlit UI
st.title("Telco Customer Churn Analysis")

df = load_data()
st.subheader("Raw Dataset")
st.write(df.head())

df = preprocess_data(df)

st.subheader("Data Visualization")
fig, ax = plt.subplots()
sns.countplot(x='Churn', data=df, ax=ax)
st.pyplot(fig)

st.subheader("Train Churn Prediction Model")
model, scaler, acc = train_model(df)
st.write(f"Model Accuracy: {acc:.2f}")

st.subheader("Predict Churn for a New Customer")
input_data = {}
for col in df.drop(columns=['Churn']).columns:
    input_data[col] = st.number_input(f"{col}", value=0.0)
input_df = pd.DataFrame([input_data])
input_df = scaler.transform(input_df)

if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    st.write("Churn Prediction:", "Yes" if prediction else "No")

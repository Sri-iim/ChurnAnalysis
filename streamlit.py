import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    return df

def preprocess_data(df):
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.fillna(0, inplace=True)
    
    df["customerID"] = df["customerID"].astype(str)  # Ensure customerID remains a string
    
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
        if df[col].nunique() < 10:
            df[col] = df[col].astype("category").cat.codes
    
    return df

df = load_data()
df = preprocess_data(df)

st.title("📊 Customer Churn Prediction App")
st.sidebar.header("Navigation")
option = st.sidebar.radio("Select an option", ["Data Overview", "Visualizations", "Model Training & Prediction"])

if option == "Data Overview":
    st.subheader("Dataset Overview")
    st.write(df.head())
    st.subheader("Summary Statistics")
    st.write(df.describe())

elif option == "Visualizations":
    st.subheader("Data Visualizations")
    col = st.selectbox("Choose a column to visualize", df.columns)
    fig = px.histogram(df, x=col, color="Churn", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Churn Count")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="Churn", ax=ax)
    st.pyplot(fig, use_container_width=True)

elif option == "Model Training & Prediction":
    st.subheader("Train a Churn Prediction Model")
    
    X = df.drop(columns=["Churn", "customerID"], errors='ignore')
    y = df["Churn"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    
    st.subheader("Make Predictions")
    sample = X_test.sample(1)
    st.write("Sample Input:", sample)
    prediction = model.predict(sample)
    st.write(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load Data
def load_data():
    df = pd.read_csv("customer_churn.csv")
    df.dropna(inplace=True)
    return df

df = load_data()

# Preprocessing
def preprocess_data(df):
    label_cols = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", 
                  "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", 
                  "PaperlessBilling", "PaymentMethod", "Churn"]
    le = LabelEncoder()
    for col in label_cols:
        df[col] = le.fit_transform(df[col])
    return df

df = preprocess_data(df)

# Train Model
def train_model(df):
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "churn_model.pkl")
    return model

model = train_model(df)

# Streamlit App
st.set_page_config(page_title="Customer Churn Analysis", layout="wide")
st.title("📊 Customer Churn Analysis Dashboard")

# Sidebar Filters
st.sidebar.header("🔍 Filter Customers")
tenure = st.sidebar.slider("Select Tenure", int(df["tenure"].min()), int(df["tenure"].max()))
monthly_charges = st.sidebar.slider("Select Monthly Charges", float(df["MonthlyCharges"].min()), float(df["MonthlyCharges"].max()))
filtered_df = df[(df["tenure"] >= tenure) & (df["MonthlyCharges"] >= monthly_charges)]

# Visualization: Customer Demographics
st.subheader("📌 Customer Demographics")
fig = px.histogram(df, x="gender", color="Churn", barmode="group")
st.plotly_chart(fig)

# Visualization: Service Usage
st.subheader("📡 Service Usage Patterns")
fig = px.histogram(df, x="InternetService", color="Churn", barmode="group")
st.plotly_chart(fig)

# Churn Prediction
st.subheader("🔮 Churn Prediction")
input_data = {col: st.sidebar.number_input(col, value=float(df[col].mean())) for col in df.columns if col not in ["customerID", "Churn"]}
input_df = pd.DataFrame([input_data])

if st.button("Predict Churn"):
    model = joblib.load("churn_model.pkl")
    prediction = model.predict(input_df)
    churn_probability = model.predict_proba(input_df)[0][1]
    st.metric(label="Churn Probability", value=f"{churn_probability*100:.2f}%")

# Additional Insights
st.subheader("📈 Additional Insights")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Churn by Contract Type")
    fig = px.pie(df, names="Contract", values="Churn", title="Churn Distribution by Contract Type")
    st.plotly_chart(fig)
with col2:
    st.subheader("Monthly Charges vs Churn")
    fig = px.scatter(df, x="MonthlyCharges", y="Churn", color="Churn", title="Monthly Charges Impact on Churn")
    st.plotly_chart(fig)

st.dataframe(filtered_df)

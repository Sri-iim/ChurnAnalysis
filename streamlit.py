import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
def load_data():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    df.drop(columns=['customerID'], inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    df['tenure'] = df['tenure'].astype(int)
    return df

df = load_data()

# Streamlit UI
st.title("Telco Customer Churn Analysis")

if st.checkbox("Show Raw Data"):
    st.write(df.head())

# Data Visualization
st.subheader("Churn Distribution")
fig, ax = plt.subplots()
sns.countplot(x='Churn', data=df, ax=ax)
st.pyplot(fig)

# Data Preprocessing
df = pd.get_dummies(df, drop_first=True)
X = df.drop(columns=['Churn_Yes'])
y = df['Churn_Yes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Selection
model_options = {"Logistic Regression": LogisticRegression(),
                 "Random Forest": RandomForestClassifier()}
selected_model = st.selectbox("Select a Model", list(model_options.keys()))

if st.button("Train Model"):
    model = model_options[selected_model]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"**Accuracy:** {acc:.4f}")
    st.text(classification_report(y_test, y_pred))
    
    if selected_model in ["Random Forest", "XGBoost"]:
        st.subheader("Feature Importance")
        feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        fig, ax = plt.subplots()
        sns.barplot(x=feature_importance, y=feature_importance.index, ax=ax)
        st.pyplot(fig)

# User Prediction
st.subheader("Make Predictions")
user_input = {col: st.number_input(col, value=0.0) for col in X.columns}
if st.button("Predict Churn"):
    input_data = pd.DataFrame([user_input])
    input_data = scaler.transform(input_data)
    prediction = model_options[selected_model].predict(input_data)
    st.write("Prediction: Churn" if prediction[0] == 1 else "Prediction: No Churn")

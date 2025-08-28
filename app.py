import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# Load trained pipeline
# ======================
model = joblib.load("heart_pipeline.pkl")

st.title("❤️ Heart Disease Prediction App")

st.write("Enter patient information below:")

# ======================
# User Inputs
# ======================
age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (cp)", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", 80, 200, 120)
chol = st.number_input("Serum Cholesterol (chol)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.selectbox("Resting ECG (restecg)", [0,1,2])
thalach = st.number_input("Max Heart Rate Achieved (thalach)", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope (slope)", [0,1,2])
ca = st.number_input("Number of Major Vessels (ca)", 0, 4, 0)
thal = st.selectbox("Thal (thal)", [0,1,2,3])

# ======================
# Prediction
# ======================
if st.button("Predict"):
    # make DataFrame for single input
    input_data = pd.DataFrame({
        "age":[age],
        "sex":[1 if sex=="Male" else 0],
        "cp":[cp],
        "trestbps":[trestbps],
        "chol":[chol],
        "fbs":[fbs],
        "restecg":[restecg],
        "thalach":[thalach],
        "exang":[exang],
        "oldpeak":[oldpeak],
        "slope":[slope],
        "ca":[ca],
        "thal":[thal]
    })
    
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High risk of Heart Disease (probability: {proba:.2f})")
    else:
        st.success(f"✅ Low risk of Heart Disease (probability: {proba:.2f})")

# ======================
# Bonus: Data Visualization
# ======================
st.subheader("📊 Heart Disease Trends (Dataset Sample)")
uploaded = st.file_uploader("Upload Heart Dataset (CSV)", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    st.write(df.head())

    st.write("### Cholesterol Distribution by Target")
    plt.figure(figsize=(8,5))
    sns.boxplot(x="target", y="chol", data=df)
    st.pyplot(plt)

    st.write("### Age Histogram")
    plt.figure(figsize=(8,5))
    df["age"].hist(bins=20, edgecolor="black")
    plt.xlabel("Age")
    plt.ylabel("Count")
    st.pyplot(plt)

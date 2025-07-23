import streamlit as st
import pandas as pd
import joblib
import base64

# ========= Set background image =========
def set_bg(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url('data:image/png;base64,{encoded}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_bg("theme.png")  # replace with your image

# ========= Page config =========
st.set_page_config(page_title="Employee Salary Prediction", page_icon="💸", layout="wide")

# ========= Load trained model =========
model = joblib.load("model.pkl")

# ========= Header =========
st.markdown("""
    <h1 style='text-align: center; background: linear-gradient(90deg, #00C9FF, #92FE9D); padding: 20px; border-radius: 10px; color: black;'>
        💼 Employee Salary Prediction App
    </h1>
""", unsafe_allow_html=True)
st.markdown("### 📊 Predict Employee Salary")

# ========= Sidebar Inputs =========
with st.sidebar:
    st.header("📝 Input Employee Details")
    age = st.slider("Age", 18, 65, 24)
    fnlwgt = st.number_input("Final Weight", min_value=0, value=100000)
    education = st.selectbox("Education Level", [
        "Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"
    ])
    educational_num = st.slider("Education Num", 1, 16, 10)
    marital_status = st.selectbox("Marital Status", ["Never-married", "Married-civ-spouse", "Divorced", "Separated", "Widowed", "Married-spouse-absent"])
    occupation = st.selectbox("Occupation", ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"])
    relationship = st.selectbox("Relationship", ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"])
    race = st.selectbox("Race", ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
    capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
    hours_per_week = st.slider("Hours per week", 1, 80, 40)
    native_country = st.selectbox("Native Country", ["United-States", "Mexico", "Philippines", "Germany", "Canada", "India", "Others"])
    workclass = st.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"])

# ========= Input DataFrame =========
input_df = pd.DataFrame({
    'age': [age],
    'fnlwgt': [fnlwgt],
    'education': [education],
    'educational-num': [educational_num],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country],
    'workclass': [workclass]
})

st.write("🔎 **Input Data Preview**")
st.dataframe(input_df, use_container_width=True)

# ========= Prediction Button =========
if st.button("💡 Calculate Salary"):
    prediction = model.predict(input_df)
    st.success(f"✅ Predicted Salary Class: {prediction[0]}")

# ========= Batch Prediction =========
st.markdown("---")
st.markdown("📂 **Batch Prediction**")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("🔍 Uploaded Data Preview")
    st.dataframe(batch_data.head(), use_container_width=True)
    batch_preds = model.predict(batch_data)
    batch_data['PredictedClass'] = batch_preds
    st.write("✅ **Predictions**")
    st.dataframe(batch_data.head(), use_container_width=True)
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')

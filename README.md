# 🧠 Employee Salary Prediction

This project predicts employee salary based on demographic and job-related features using machine learning. It includes a machine learning pipeline with data preprocessing, model training, and deployment using Flask API and Streamlit web interface.

## 📁 Project Structure

📦Employee Salary Prediction
┣ 📜Employee Salary Prediction.ipynb
┣ 📜model.pkl
┣ 📜requirements.txt
┣ 📜app.py (Flask API)
┣ 📜streamlit_app.py (Web UI)
┗ 📜README.md



## 📊 Dataset

- **Source**: UCI Adult Income dataset
- **File Used**: `adult 3.csv`
- **Target Variable**: Income (<=50K or >50K)

### Features:
- Age
- Workclass
- Education
- Marital Status
- Occupation
- Race
- Sex
- Hours per week
- Native country
- etc.

## 🔍 Tech Stack

- **Python** (Pandas, NumPy, Matplotlib, Seaborn)
- **Machine Learning**: Scikit-learn, Statsmodels
- **Model Deployment**: Flask REST API
- **Web Interface**: Streamlit
- **Serialization**: `pickle` and `joblib`

## ⚙️ How It Works

### 1. Data Preprocessing
- Missing values handled
- Label Encoding / One-Hot Encoding used
- Feature scaling with `StandardScaler`

### 2. Model Building
- Logistic Regression and other ML algorithms trained
- Multicollinearity checked using VIF
- Best performing model saved using `pickle`

### 3. API (Flask)
- Accepts JSON input
- Returns predicted salary category
- Run using: `python app.py`

### 4. Web UI (Streamlit)
- Easy-to-use interface to predict salary
- Run using: `streamlit run streamlit_app.py`

## 🚀 Getting Started

### 🔧 Prerequisites

Install dependencies:
```bash
pip install -r requirements.txt
▶️ Run Flask API
bash

python app.py
🌐 Run Streamlit App
bash

streamlit run streamlit_app.py
📦 Model Input Format
json

{
  "age": 39,
  "workclass": "State-gov",
  "education": "Bachelors",
  "marital_status": "Never-married",
  "occupation": "Adm-clerical",
  "relationship": "Not-in-family",
  "race": "White",
  "sex": "Male",
  "hours_per_week": 40,
  "native_country": "United-States"
}
📈 Model Evaluation
Confusion Matrix

Accuracy Score

Classification Report

Cross-validation scores


📌 To-Do
Add support for more ML models (RandomForest, XGBoost)



Host the app on Ngrok/Render/Heroku

🧑‍💻 Author
Your Name – Swaraj Dara



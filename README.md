# 🫀 Heart Disease Prediction App

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

> A machine learning web application that predicts the likelihood of heart disease based on patient health data, built with **Streamlit** and deployable via **Ngrok** or **Streamlit Cloud**.

---

## 📌 Overview

Cardiovascular disease is the leading cause of death globally. Early detection can drastically improve patient outcomes. This project trains and compares multiple classification models on the **UCI Heart Disease dataset**, then serves the best-performing model through an interactive web interface where users can enter their health parameters and get an instant prediction.

---

## ✨ Features

- 🖥️ Interactive **Streamlit UI** for real-time user input and prediction
- 📊 Data visualization of heart disease trends and feature distributions
- 🔁 ML **pipeline** with preprocessing, feature selection, and model inference
- 🌐 Deployable locally or publicly via **Ngrok** / **Streamlit Cloud**
- 💾 Trained model saved as a `.pkl` pipeline for easy reuse

---

## 📂 Project Structure

```
heart-disease-prediction/
│
├── app.py                          # Streamlit web app
├── heart_pipeline.py               # Model training and pipeline building
├── heart_pipeline.pkl              # Saved trained Random Forest pipeline
├── requirements.txt                # Python dependencies
├── .gitignore
│
├── data/                           # Dataset files
│   └── heart.csv                   # UCI Heart Disease dataset
│
├── notebooks/                      # Jupyter notebooks for EDA & modeling
│   └── heart_prediction_disease_v3.2-beta.2.zip
│
├── models/                         # Saved model artifacts
│
├── results/                        # Evaluation metrics output
│   └── metrics.csv
│
└── deployment/                     # Ngrok deployment instructions
```

---

## 🗂️ Dataset

**Source:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)

| Feature | Type | Description |
|---------|------|-------------|
| `age` | int | Age of the patient |
| `sex` | binary | 1 = male, 0 = female |
| `cp` | int | Chest pain type (0–3) |
| `trestbps` | int | Resting blood pressure (mm Hg) |
| `chol` | int | Serum cholesterol (mg/dl) |
| `fbs` | binary | Fasting blood sugar > 120 mg/dl |
| `restecg` | int | Resting ECG results (0–2) |
| `thalach` | int | Maximum heart rate achieved |
| `exang` | binary | Exercise-induced angina |
| `oldpeak` | float | ST depression induced by exercise |
| `slope` | int | Slope of the peak exercise ST segment |
| `ca` | int | Number of major vessels colored by fluoroscopy |
| `thal` | int | Thalassemia type |
| `target` | binary | **1 = Heart disease present, 0 = Absent** |

---

## 🔬 Methodology

### Preprocessing
- Missing value imputation
- Standard scaling for numerical features
- One-hot encoding for categorical features

### Feature Selection
- **RFE** (Recursive Feature Elimination)
- **Chi-Square test** for categorical feature relevance

### Models Compared
| Model | Notes |
|-------|-------|
| Logistic Regression | Linear baseline |
| Decision Tree | Rule-based |
| Random Forest | ✅ Final model (best accuracy) |
| SVM | Support vector classification |

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- AUC-ROC Curve

All metrics saved to `results/metrics.csv`.

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/MoHussein197/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Run locally
```bash
streamlit run app.py
```
Then open your browser at `http://localhost:8501`

### Deploy publicly with Ngrok
Follow the instructions in the `deployment/` folder to get a shareable public URL.

---

## 📦 Dependencies

```
streamlit
scikit-learn
pandas
numpy
matplotlib
seaborn
joblib
pyngrok
```

---

## 📝 Notes

- Ensure Python has **write permissions** on the `results/` folder to save metrics.
- If Logistic Regression shows a convergence warning, increase `max_iter` in `heart_pipeline.py` or ensure features are properly scaled.
- The `.pkl` file must be in the root directory for `app.py` to load the model correctly.

---

## 👤 Author

**Mohamed Hussein Kamal**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/mohamed-hussein)
[![GitHub](https://img.shields.io/badge/GitHub-MoHussein197-black?logo=github)](https://github.com/MoHussein197)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🔗 References

1. UCI Machine Learning Repository — [Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)
2. Scikit-learn Documentation — https://scikit-learn.org
3. Streamlit Documentation — https://docs.streamlit.io

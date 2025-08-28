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
├── heart_pipeline.py           # Main ML pipeline and app logic
├── heart_pipeline.pkl          # Saved trained Random Forest model
├── requirements.txt            # Python dependencies
├── README.md                   # This README file
├── results/                    # Folder for evaluation metrics
│   └── evaluation_metrics.txt
├── ngrok_setup.txt             # Instructions to set up Ngrok for deployment
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
pip install -r requirements.txt
```

---

## 🏃 How to Run

1. **Run the Streamlit app**

```bash
streamlit run heart_pipeline.py
```

2. **Optional: Deploy with Ngrok**

* Follow instructions in `ngrok_setup.txt`
* Run the command provided to get a public URL for your app

---

## Model & Pipeline

* **Dataset:** UCI Heart Disease dataset
* **Preprocessing:** Scaling, one-hot encoding for categorical features
* **Feature Selection:** RFE and Chi-Square test
* **Models:** Logistic Regression, Decision Tree, Random Forest, SVM
* **Final Model:** Random Forest trained on selected features, saved as `heart_pipeline.pkl`

---

## Evaluation Metrics

* Metrics for all models are saved in `results/evaluation_metrics.txt`
* Includes: Accuracy, Precision, Recall, F1 Score, AUC (if applicable)

---

## Notes

* Ensure Python has **write permissions** for the `results/` folder to save metrics.
* If logistic regression gives a convergence warning, increase `max_iter` or scale features properly.

---

## Author

* Mohamed Hussein
* Email: [mohamedhusseineissa14@gmail.com]
* GitHub: [MoHussein197](https://github.com/MoHussein197)


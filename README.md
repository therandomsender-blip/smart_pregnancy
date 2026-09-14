# 🤰 Smart Pregnancy Monitoring System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20Random%20Forest-green.svg)](#model-performance)
[![Explainable AI](https://img.shields.io/badge/XAI-SHAP-orange.svg)](#explainable-ai-shap)

An intelligent clinical decision support interface for **monitoring and early risk prediction of pregnancy-related complications** using machine learning and Explainable AI (XAI).

Developed at **Acharya Vinoba Bhave Rural Hospital (AVBRH), Datta Meghe Institute of Higher Education and Research (DMIHER)**.

---

## 📌 Project Overview

Maternal healthcare outcomes can be significantly enhanced through early identification of high-risk pregnancies and gestational diabetes mellitus (GDM). This application empowers clinicians, obstetricians, and healthcare workers to:
- Input real-time patient clinical biomarkers and medical history.
- Obtain consensus risk predictions using **XGBoost** and **Random Forest** classifiers.
- Understand patient-specific risk drivers through **SHAP (SHapley Additive exPlanations)** waterfall and feature attribution plots.
- Receive evidence-based clinical referral and monitoring recommendations.

---

## 🩺 Clinical Biomarkers & Features

The model evaluates 10 core clinical features selected via Recursive Feature Elimination (RFE) and clinical relevance:

| Feature | Description | Range / Values |
| :--- | :--- | :--- |
| **Age** | Patient age | 15 – 50 years |
| **BMI** | Body Mass Index | 15.0 – 45.0 kg/m² |
| **Systolic BP** | Systolic Blood Pressure | 80 – 200 mmHg |
| **Diastolic BP** | Diastolic Blood Pressure | 50 – 130 mmHg |
| **Blood Sugar (BS)** | Fasting blood glucose level | 3.0 – 20.0 mmol/L |
| **Heart Rate** | Resting heart rate | 50 – 130 bpm |
| **Previous Complications**| Prior obstetric complications | Yes / No |
| **Preexisting Diabetes** | Diabetes diagnosed before pregnancy | Yes / No |
| **Gestational Diabetes** | Diabetes diagnosed during pregnancy | Yes / No |
| **Mental Health** | Documented mental health concerns | Yes / No |

---

## 🏆 Model Performance

Models were trained, validated, and evaluated on preprocessed clinical cohorts:

| Metric | XGBoost | Random Forest |
| :--- | :---: | :---: |
| **Overall Accuracy** | **94.20%** | **92.40%** |
| **High Risk F1-Score** | **0.97** | **0.93** |
| **High GDM Risk F1-Score** | **1.00** | **1.00** |
| **Low Risk F1-Score** | **0.91** | **0.89** |
| **Moderate Risk F1-Score**| **0.89** | **0.87** |

---

## 🧠 Explainable AI (SHAP)

To ensure clinical trust and interpretability:
- **Global Feature Importance**: Identifies population-level biomarkers driving risk categorization across the dataset.
- **Patient-Specific Waterfall Plots**: Quantifies how each individual biomarker increases or decreases the log-odds of a predicted risk tier.

---

## 🚀 Local Installation & Setup

### Prerequisites
- Python 3.10+ installed
- Git installed

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/<your-username>/smart_pregnancy.git
   cd smart_pregnancy
   ```

2. **Create and Activate a Virtual Environment**
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .venv\Scripts\Activate.ps1

   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit App**
   ```bash
   streamlit run app.py
   ```
   Open your browser at `http://localhost:8501`.

---

## ☁️ Deployment on Streamlit Community Cloud

Deploying this app is completely free and takes less than 2 minutes:

1. **Push to GitHub**: Make sure this repository is pushed to your personal GitHub account.
2. **Go to Streamlit Community Cloud**: Visit [share.streamlit.io](https://share.streamlit.io) and log in using your GitHub account.
3. **Create New App**:
   - Click **"Create app"** / **"New app"**.
   - Select your repository: `smart_pregnancy`.
   - Branch: `main`.
   - Main file path: `app.py`.
4. **Deploy**: Click **"Deploy!"**.
   - Streamlit Cloud will automatically read `requirements.txt`, install dependencies, and launch your application live with a public URL!

---

## 📂 Repository Structure

```
smart_pregnancy/
├── .streamlit/
│   └── config.toml                  # Streamlit UI theme and server settings
├── data/
│   └── maternal_anemia_dataset.csv.xlsx # Source clinical dataset
├── outputs/
│   ├── label_encoder_classes.json   # Class index to label mapping
│   ├── preprocessing_report.json    # Pipeline execution report
│   ├── shap_summary.png             # Global SHAP importance plot
│   ├── shap_waterfall.png           # Sample patient waterfall
│   ├── train_data.csv               # Processed training split
│   ├── val_data.csv                 # Processed validation split
│   └── test_data.csv                # Processed testing split
├── app.py                           # Main Streamlit web application
├── confusion_matrices.png           # Multi-class confusion matrix plots
├── model_training.py                # Model training and validation script
├── preprocessing_pipeline.py        # MICE, SMOTE, and normalization pipeline
├── rf_model.pkl                     # Serialized Random Forest classifier
├── xgb_model.pkl                    # Serialized XGBoost classifier
├── shap_explainability.py           # SHAP analysis script
├── requirements.txt                 # Project dependencies
├── .gitignore                       # Git ignore configuration
└── README.md                        # Documentation
```

---

## 👨‍🔬 Authors & Acknowledgments

- **Researcher / Developer**: Asad ul Hakeem (DMET1224037)
- **Academic Mentor**: Prof. (Dr.) K.T.V. Reddy
- **Affiliation**: Datta Meghe Institute of Higher Education and Research (DMIHER), AVBRH

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

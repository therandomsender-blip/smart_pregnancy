# 🤰 Smart Pregnancy Monitoring System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://therandomsender-blip-smart-pregnancy-app-ggocij.streamlit.app/)
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

## 🏆 Model Performance & 95% Confidence Intervals

### 🔬 Scientifically Primary Model (Leakage-Free Predictive Model)
> **Scientific Design**: To prevent target leakage, **Gestational Diabetes is excluded as an input feature**. The model predicts GDM and obstetric risk purely from physiological biomarkers (`Blood Sugar (BS)`, `BMI`, `Systolic/Diastolic BP`, `Heart Rate`, `Age`, `Preexisting Diabetes`, `Previous Complications`, `Mental Health`).

Evaluated on the independent test cohort ($N = 448$). All confidence intervals ($95\%\text{ CI}$) were calculated via **non-parametric bootstrapping ($2,000$ iterations)**:

| Metric | XGBoost (Scientifically Primary) | XGBoost (95% CI) | Random Forest | Random Forest (95% CI) |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | **91.07%** | **88.39% – 93.53%** | **88.62%** | **85.71% – 91.52%** |
| **Macro F1-Score** | **0.9108** | **0.8842 – 0.9359** | **0.8863** | **0.8566 – 0.9141** |
| **ROC-AUC (Weighted OvR)** | **0.9879** | **0.9819 – 0.9929** | **0.9812** | **0.9728 – 0.9880** |
| **🟠 High GDM F1-Score** | **0.9600** | **0.9309 – 0.9829** | **0.9515** | **0.9212 – 0.9762** |
| **— GDM Recall (Sensitivity)**| **96.43%** | **92.38% – 99.18%** | **96.43%** | **92.56% – 99.15%** |
| **— GDM Precision** | **95.58%** | **91.41% – 99.08%** | **93.91%** | **89.19% – 97.70%** |
| **🔴 High Risk F1-Score** | **0.9279** | **0.8898 – 0.9607** | **0.9091** | **0.8652 – 0.9459** |
| **🟢 Low Risk F1-Score** | **0.8796** | **0.8309 – 0.9224** | **0.8473** | **0.7895 – 0.8984** |
| **🟡 Moderate Risk F1-Score**| **0.8755** | **0.8295 – 0.9182** | **0.8374** | **0.7851 – 0.8828** |

---

### 📊 Ablation Analysis: Leakage-Free vs. Diagnostic Flag
In clinical research, distinguishing between risk screening and known prior diagnosis is critical:

| Evaluation Protocol | Overall Accuracy | GDM F1-Score | GDM Sensitivity | Clinical Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Scientifically Primary (Biomarkers Only)** | **91.07%** | **0.9600** | **96.43%** | **Early prediction & screening prior to GDM onset** |
| Benchmark with Prior Diagnosis Flag | 94.20% | 1.0000 | 100.00% | Consensus triaging when GDM is already formally charted |

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
   git clone https://github.com/therandomsender-blip/smart_pregnancy.git
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
   - Select your repository: `therandomsender-blip/smart_pregnancy`.
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

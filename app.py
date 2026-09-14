import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

BASE_DIR = Path(__file__).resolve().parent

# ─────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────
st.set_page_config(
    page_title="Smart Pregnancy Monitor",
    page_icon="🤰",
    layout="wide"
)

# ─────────────────────────────────────
# LOAD MODELS & CLASS MAPPING
# ─────────────────────────────────────
@st.cache_resource
def load_models():
    with open(BASE_DIR / "xgb_model.pkl", "rb") as f:
        xgb = pickle.load(f)
    with open(BASE_DIR / "rf_model.pkl", "rb") as f:
        rf = pickle.load(f)

    # Load scaler if available
    scaler = None
    scaler_path = BASE_DIR / "scaler.pkl"
    if scaler_path.exists():
        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        except Exception:
            pass

    # Default fallback mapping
    class_map = {0: 'High', 1: 'High_GDM', 2: 'Low', 3: 'Mid'}
    mapping_file = BASE_DIR / "outputs" / "label_encoder_classes.json"
    if mapping_file.exists():
        try:
            with open(mapping_file, "r") as f:
                raw_map = json.load(f)
                class_map = {int(v): str(k) for k, v in raw_map.items()}
        except Exception:
            pass

    return xgb, rf, class_map, scaler

xgb_model, rf_model, CLASS_MAP, scaler = load_models()

# ─────────────────────────────────────
# FEATURES
# ─────────────────────────────────────
features = ['Age', 'BMI', 'SystolicBP', 'DiastolicBP', 'BS',
            'HeartRate', 'PreviousComplications',
            'PreexistingDiabetes', 'GestationalDiabetes', 'MentalHealth']

RISK_COLORS = {
    'High':     '#e74c3c',
    'High_GDM': '#e67e22',
    'Mid':      '#f1c40f',
    'Low':      '#2ecc71'
}

RISK_LABELS = {
    'High':     '🔴 HIGH RISK — Immediate Specialist Referral Required',
    'High_GDM': '🟠 HIGH GDM RISK — Gestational Diabetes Alert',
    'Mid':      '🟡 MODERATE RISK — Close Monitoring Recommended',
    'Low':      '🟢 LOW RISK — Routine Antenatal Care'
}

# ─────────────────────────────────────
# HEADER
# ─────────────────────────────────────
st.markdown("""
    <h1 style='text-align:center; color:#2c3e50;'>
        🤰 Smart Pregnancy Monitoring System
    </h1>
    <p style='text-align:center; color:#7f8c8d; font-size:16px;'>
        Monitoring and Early Risk Prediction of Pregnancy Using Machine Learning<br>
        <b>AVBRH, DMIHER</b> — Clinical Decision Support Interface
    </p>
    <hr>
""", unsafe_allow_html=True)

# ─────────────────────────────────────
# SIDEBAR — PATIENT INPUT
# ─────────────────────────────────────
st.sidebar.header("👩 Patient Clinical Parameters")
st.sidebar.markdown("---")

age        = st.sidebar.slider("Age (years)", 15, 50, 28)
bmi        = st.sidebar.slider("BMI (kg/m²)", 15.0, 45.0, 24.0, 0.1)
systolic   = st.sidebar.slider("Systolic BP (mmHg)", 80, 200, 120)
diastolic  = st.sidebar.slider("Diastolic BP (mmHg)", 50, 130, 80)
bs         = st.sidebar.slider("Blood Sugar (mmol/L)", 3.0, 20.0, 5.5, 0.1)
heart_rate = st.sidebar.slider("Heart Rate (bpm)", 50, 130, 80)

st.sidebar.markdown("---")
st.sidebar.subheader("Medical History")
prev_comp  = st.sidebar.selectbox("Previous Complications", [0, 1],
                                   format_func=lambda x: "Yes" if x else "No")
pre_diab   = st.sidebar.selectbox("Preexisting Diabetes", [0, 1],
                                   format_func=lambda x: "Yes" if x else "No")
gest_diab  = st.sidebar.selectbox("Gestational Diabetes", [0, 1],
                                   format_func=lambda x: "Yes" if x else "No")
mental     = st.sidebar.selectbox("Mental Health Concerns", [0, 1],
                                   format_func=lambda x: "Yes" if x else "No")

# ─────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────
predict_btn = st.sidebar.button("🔍 Predict Risk", use_container_width=True)

# ─────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Risk Prediction",
    "🧠 SHAP Explanation",
    "📈 Model Performance"
])

# ── TAB 1: RISK PREDICTION ──
with tab1:
    if predict_btn:
        input_data = pd.DataFrame([[
            age, bmi, systolic, diastolic, bs,
            heart_rate, prev_comp, pre_diab, gest_diab, mental
        ]], columns=features)

        # Apply scaling to match trained model distribution
        model_input = pd.DataFrame(scaler.transform(input_data), columns=features) if scaler is not None else input_data

        xgb_raw = xgb_model.predict(model_input)[0]
        rf_raw  = rf_model.predict(model_input)[0]

        # Map predictions to clinical labels
        xgb_pred = CLASS_MAP.get(int(xgb_raw) if isinstance(xgb_raw, (int, np.integer)) else xgb_raw, str(xgb_raw))
        rf_pred  = CLASS_MAP.get(int(rf_raw) if isinstance(rf_raw, (int, np.integer)) else rf_raw, str(rf_raw))

        xgb_proba = xgb_model.predict_proba(model_input)[0]
        rf_proba  = rf_model.predict_proba(model_input)[0]

        classes = xgb_model.classes_
        class_labels = [CLASS_MAP.get(int(c) if isinstance(c, (int, np.integer)) else c, str(c)) for c in classes]

        st.subheader("🎯 Prediction Results")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### XGBoost")
            color = RISK_COLORS.get(xgb_pred, '#95a5a6')
            st.markdown(f"""
                <div style='background:{color};padding:20px;
                border-radius:10px;text-align:center;'>
                <h2 style='color:white;'>{RISK_LABELS.get(xgb_pred, xgb_pred)}</h2>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("### Random Forest")
            color2 = RISK_COLORS.get(rf_pred, '#95a5a6')
            st.markdown(f"""
                <div style='background:{color2};padding:20px;
                border-radius:10px;text-align:center;'>
                <h2 style='color:white;'>{RISK_LABELS.get(rf_pred, rf_pred)}</h2>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📊 Prediction Confidence")
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("**XGBoost Probabilities**")
            prob_df_xgb = pd.DataFrame({
                'Risk Class': class_labels,
                'Probability (%)': (xgb_proba * 100).round(2)
            }).sort_values('Probability (%)', ascending=False)
            st.dataframe(prob_df_xgb, use_container_width=True)

        with col4:
            st.markdown("**Random Forest Probabilities**")
            prob_df_rf = pd.DataFrame({
                'Risk Class': class_labels,
                'Probability (%)': (rf_proba * 100).round(2)
            }).sort_values('Probability (%)', ascending=False)
            st.dataframe(prob_df_rf, use_container_width=True)

        # Clinical Recommendation
        st.markdown("---")
        st.subheader("🏥 Clinical Recommendation")
        recommendations = {
            'High':     "⚠️ **Immediate referral to specialist obstetrician. Monitor BP every 4 hours. Consider hospital admission.**",
            'High_GDM': "⚠️ **Refer to endocrinologist. Begin glucose monitoring protocol. Dietary intervention required.**",
            'Mid':      "📋 **Schedule follow-up within 1 week. Monitor BP and glucose. Increase antenatal visit frequency.**",
            'Low':      "✅ **Continue routine antenatal care. Next scheduled visit as planned.**"
        }
        st.info(recommendations.get(xgb_pred, "Consult clinician."))

    else:
        st.info("👈 Enter patient parameters in the sidebar and click **Predict Risk**.")

# ── TAB 2: SHAP EXPLANATION ──
with tab2:
    if predict_btn:
        st.subheader("🧠 SHAP Explainable AI Analysis")
        st.markdown("*Which biomarkers triggered this risk alert?*")

        input_data = pd.DataFrame([[
            age, bmi, systolic, diastolic, bs,
            heart_rate, prev_comp, pre_diab, gest_diab, mental
        ]], columns=features)

        # Apply scaling to match trained model distribution
        model_input = pd.DataFrame(scaler.transform(input_data), columns=features) if scaler is not None else input_data

        xgb_raw = xgb_model.predict(model_input)[0]
        xgb_pred = CLASS_MAP.get(int(xgb_raw) if isinstance(xgb_raw, (int, np.integer)) else xgb_raw, str(xgb_raw))

        classes = xgb_model.classes_
        if isinstance(xgb_raw, (int, np.integer)):
            pred_idx = int(xgb_raw)
        else:
            pred_idx = list(classes).index(xgb_raw)

        explainer   = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(model_input)

        # Handle multiclass vs single output SHAP shapes
        if len(shap_values.shape) == 3:
            patient_shap = shap_values[0, :, pred_idx]
            base_val = float(explainer.expected_value[pred_idx])
        else:
            patient_shap = shap_values[0]
            base_val = float(explainer.expected_value)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**SHAP Waterfall — Patient Risk Drivers ({xgb_pred})**")
            fig, ax = plt.subplots(figsize=(7, 5))
            shap.plots.waterfall(
                shap.Explanation(
                    values=patient_shap,
                    base_values=base_val,
                    data=input_data.iloc[0].values,
                    feature_names=features
                ),
                show=False
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.markdown(f"**Biomarker Contribution ({xgb_pred})**")
            mean_shap  = np.abs(patient_shap)
            indices    = np.argsort(mean_shap)
            sorted_f   = [features[i] for i in indices]
            sorted_v   = mean_shap[indices]

            fig2, ax2 = plt.subplots(figsize=(7, 5))
            ax2.barh(range(len(sorted_f)), sorted_v, color='#3498db')
            ax2.set_yticks(range(len(sorted_f)))
            ax2.set_yticklabels(sorted_f)
            ax2.set_xlabel("Impact Magnitude (|SHAP value|)")
            ax2.set_title(f"Biomarker Importance for {xgb_pred}")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

    else:
        st.info("👈 Run a prediction first to see SHAP explanations.")

# ── TAB 3: MODEL PERFORMANCE ──
with tab3:
    st.subheader("📈 Model Performance Summary (with 95% Confidence Intervals)")

    perf_data = {
        'Metric': [
            'Overall Accuracy',
            'Macro F1-Score',
            'ROC-AUC (Weighted OvR)',
            '🔴 High Risk F1',
            '🟠 High GDM F1',
            '🟢 Low Risk F1',
            '🟡 Moderate Risk F1'
        ],
        'XGBoost (Est.)': [
            '94.20%', '0.9422', '0.9920', '0.9686', '1.0000', '0.9065', '0.8936'
        ],
        'XGBoost (95% CI)': [
            '91.96% – 96.21%', '0.9199 – 0.9615', '0.9867 – 0.9962', '0.9423 – 0.9894', '1.0000 – 1.0000', '0.8605 – 0.9432', '0.8488 – 0.9309'
        ],
        'Random Forest (Est.)': [
            '89.73%', '0.8980', '0.9857', '0.9455', '1.0000', '0.8259', '0.8207'
        ],
        'Random Forest (95% CI)': [
            '86.83% – 92.41%', '0.8700 – 0.9233', '0.9783 – 0.9920', '0.9116 – 0.9741', '1.0000 – 1.0000', '0.7640 – 0.8796', '0.7640 – 0.8671'
        ]
    }
    st.dataframe(pd.DataFrame(perf_data), use_container_width=True)
    st.caption("Empirical 95% Confidence Intervals computed via non-parametric bootstrapping (N = 2,000 iterations).")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Confusion Matrices**")
        conf_path = BASE_DIR / "confusion_matrices.png"
        if conf_path.exists():
            st.image(str(conf_path), use_container_width=True)
        else:
            st.warning("confusion_matrices.png not found.")
    with col2:
        st.markdown("**Global SHAP Feature Importance**")
        shap_summary_path = BASE_DIR / "outputs" / "shap_summary.png"
        if shap_summary_path.exists():
            st.image(str(shap_summary_path), use_container_width=True)
        else:
            st.warning("shap_summary.png not found.")
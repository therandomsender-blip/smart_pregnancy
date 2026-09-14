import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# ─────────────────────────────────────
# 1. LOAD MODEL AND DATA
# ─────────────────────────────────────
print("\nSTEP 1 — Loading model and test data")
print("=" * 60)

with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

test = pd.read_csv("outputs/test_data.csv")

features = ['Age', 'BMI', 'SystolicBP', 'DiastolicBP', 'BS',
            'HeartRate', 'PreviousComplications',
            'PreexistingDiabetes', 'GestationalDiabetes', 'MentalHealth']

X_test = test[features]

# ─────────────────────────────────────
# 2. SHAP EXPLAINER
# ─────────────────────────────────────
print("\nSTEP 2 — Generating SHAP values")
print("=" * 60)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# ─────────────────────────────────────
# 3. SUMMARY PLOT (Global Feature Importance)
# ─────────────────────────────────────
print("\nSTEP 3 — Saving SHAP Summary Plot")
print("=" * 60)

plt.figure()
shap.summary_plot(shap_values, X_test,
                  class_names=['High', 'High_GDM', 'Low', 'Mid'],
                  show=False)
plt.tight_layout()
plt.savefig("outputs/shap_summary.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/shap_summary.png")

# ─────────────────────────────────────
# 4. WATERFALL PLOT (Single Patient)
# ─────────────────────────────────────
print("\nSTEP 4 — Saving SHAP Waterfall Plot (Patient 0)")
print("=" * 60)

plt.figure()
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[0][0],
        base_values=explainer.expected_value[0],
        data=X_test.iloc[0],
        feature_names=features
    ),
    show=False
)
plt.tight_layout()
plt.savefig("outputs/shap_waterfall.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/shap_waterfall.png")

# ─────────────────────────────────────
# 5. BAR PLOT (Mean Feature Importance)
# ─────────────────────────────────────
print("\nSTEP 5 — Saving SHAP Bar Plot")
print("=" * 60)

import numpy as np
mean_shap = np.abs(shap_values).mean(axis=(0, 1))
indices = np.argsort(mean_shap)
sorted_features = [features[i] for i in indices]
sorted_values = mean_shap[indices]

plt.figure(figsize=(8, 5))
plt.barh(range(len(sorted_features)), sorted_values, color='steelblue')
plt.yticks(range(len(sorted_features)), sorted_features)
plt.xlabel("Mean |SHAP Value|")
plt.title("Global Feature Importance (SHAP)")
plt.tight_layout()
plt.savefig("outputs/shap_bar.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/shap_bar.png")

print("\nSHAP COMPLETE")
print("=" * 60)
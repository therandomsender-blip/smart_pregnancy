import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────
print("\nSTEP 1 — Loading preprocessed data")
print("=" * 60)

train = pd.read_csv("outputs/train_data.csv")
val   = pd.read_csv("outputs/val_data.csv")
test  = pd.read_csv("outputs/test_data.csv")

features = ['Age', 'BMI', 'SystolicBP', 'DiastolicBP', 'BS',
            'HeartRate', 'PreviousComplications',
            'PreexistingDiabetes', 'GestationalDiabetes', 'MentalHealth']

X_train = train[features]
y_train = train["CombinedRisk_enc"]

X_val   = val[features]
y_val   = val["CombinedRisk_enc"]

X_test  = test[features]
y_test  = test["CombinedRisk_enc"]

print(f"  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

# ─────────────────────────────────────
# 2. RANDOM FOREST
# ─────────────────────────────────────
print("\nSTEP 2 — Training Random Forest")
print("=" * 60)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
rf.fit(X_train, y_train)

rf_val_acc = accuracy_score(y_val, rf.predict(X_val))
print(f"  Random Forest Val Accuracy: {rf_val_acc * 100:.2f}%")

# ─────────────────────────────────────
# 3. XGBOOST
# ─────────────────────────────────────
print("\nSTEP 3 — Training XGBoost")
print("=" * 60)

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
xgb.fit(X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False)

xgb_val_acc = accuracy_score(y_val, xgb.predict(X_val))
print(f"  XGBoost Val Accuracy: {xgb_val_acc * 100:.2f}%")

# ─────────────────────────────────────
# 4. FINAL EVALUATION ON TEST SET
# ─────────────────────────────────────
print("\nSTEP 4 — Test Set Evaluation")
print("=" * 60)

labels = ['High', 'High_GDM', 'Low', 'Mid']

print("\n  --- Random Forest ---")
rf_test_pred = rf.predict(X_test)
print(f"  Test Accuracy: {accuracy_score(y_test, rf_test_pred) * 100:.2f}%")
print(classification_report(y_test, rf_test_pred, target_names=labels))

print("\n  --- XGBoost ---")
xgb_test_pred = xgb.predict(X_test)
print(f"  Test Accuracy: {accuracy_score(y_test, xgb_test_pred) * 100:.2f}%")
print(classification_report(y_test, xgb_test_pred, target_names=labels))

# ─────────────────────────────────────
# 5. CONFUSION MATRIX PLOTS
# ─────────────────────────────────────
print("\nSTEP 5 — Saving Confusion Matrices")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, preds, title in zip(axes,
                             [rf_test_pred, xgb_test_pred],
                             ["Random Forest", "XGBoost"]):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f"{title} — Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
print("  Saved: confusion_matrices.png")

# ─────────────────────────────────────
# 6. SAVE MODELS
# ─────────────────────────────────────
print("\nSTEP 6 — Saving Models")
print("=" * 60)

with open("rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)

with open("xgb_model.pkl", "wb") as f:
    pickle.dump(xgb, f)

print("  Saved: rf_model.pkl")
print("  Saved: xgb_model.pkl")
print("\nMODEL TRAINING COMPLETE")
print("=" * 60)
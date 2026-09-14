"""
Smart Pregnancy Monitoring System
Data Preprocessing Pipeline
Student: Asad ul Hakeem | DMET1224037 | DMIHER
Mentor: Prof. (Dr.) K.T.V. Reddy

Pipeline Steps:
  1. Data Loading & Inspection
  2. Label Standardisation (RiskLevel + AnemiaSeverity → combined target)
  3. Categorical Encoding
  4. MICE Imputation (missing values)
  5. SMOTE (class imbalance)
  6. Min-Max Normalisation
  7. RFE Feature Selection
  8. Train / Validation / Test Split (80:10:10)
  9. Save outputs + preprocessing report
"""

import pandas as pd
import numpy as np
import warnings
import json
from pathlib import Path

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

report = {}   # collects stats for the final report

# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Loading Dataset")
print("=" * 60)
df = pd.read_excel("data/maternal_anemia_dataset.csv.xlsx")
report["original_shape"] = list(df.shape)
print(f"  Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────────
# 2. LABEL STANDARDISATION + COMBINED TARGET
# ─────────────────────────────────────────────
print("\nSTEP 2 — Label Standardisation & Combined Target")
print("=" * 60)

# Standardise RiskLevel (mixed casing / spacing issue)
risk_map = {
    "Low": "Low", "low risk": "Low",
    "High": "High", "high risk": "High",
    "mid risk": "Mid"
}
df["RiskLevel"] = df["RiskLevel"].map(risk_map)
print(f"  RiskLevel after standardisation:\n{df['RiskLevel'].value_counts().to_string()}")

# Build combined target: encode GDM flag into risk level
# Logic:
#   GestationalDiabetes == 1  → bump to High regardless
#   Otherwise keep standardised RiskLevel
def combined_target(row):
    if pd.isna(row["RiskLevel"]):
        return np.nan
    gdm = row.get("GestationalDiabetes", 0)
    if gdm == 1.0:
        return "High_GDM"
    return row["RiskLevel"]

df["CombinedRisk"] = df.apply(combined_target, axis=1)

# Drop rows where target is still NaN (18 rows)
before = len(df)
df = df.dropna(subset=["CombinedRisk"])
report["rows_dropped_no_target"] = before - len(df)
print(f"\n  Combined target distribution:\n{df['CombinedRisk'].value_counts().to_string()}")
print(f"\n  Dropped {before - len(df)} rows with missing target.")

# ─────────────────────────────────────────────
# 3. CATEGORICAL ENCODING
# ─────────────────────────────────────────────
print("\nSTEP 3 — Categorical Encoding")
print("=" * 60)

# Binary Yes/No columns
binary_cols = ["Fatigue", "Dizziness"]
for col in binary_cols:
    df[col] = df[col].map({"Yes": 1, "No": 0})
    print(f"  {col}: Yes→1, No→0")

# Ordinal: NutritionStatus
nutrition_map = {"Poor": 0, "Average": 1, "Good": 2}
df["NutritionStatus"] = df["NutritionStatus"].map(nutrition_map)
print("  NutritionStatus: Poor→0, Average→1, Good→2")

# Ordinal: AnemiaSeverity (clinically ordered)
anemia_map = {"Normal": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
df["AnemiaSeverity"] = df["AnemiaSeverity"].map(anemia_map)
print("  AnemiaSeverity: Normal→0, Mild→1, Moderate→2, Severe→3")

# Encode target label
le = LabelEncoder()
df["CombinedRisk_enc"] = le.fit_transform(df["CombinedRisk"])
print(f"\n  Target classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")
report["target_classes"] = {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}

# ─────────────────────────────────────────────
# 4. MICE IMPUTATION
# ─────────────────────────────────────────────
print("\nSTEP 4 — MICE Imputation")
print("=" * 60)

feature_cols = [c for c in df.columns if c not in ["RiskLevel", "CombinedRisk", "CombinedRisk_enc"]]

missing_before = df[feature_cols].isnull().sum()
report["missing_before"] = missing_before[missing_before > 0].to_dict()
print("  Missing values before MICE:")
print(missing_before[missing_before > 0].to_string())

mice = IterativeImputer(random_state=42, max_iter=10)
df_imputed = pd.DataFrame(
    mice.fit_transform(df[feature_cols]),
    columns=feature_cols
)

missing_after = df_imputed.isnull().sum().sum()
report["missing_after"] = int(missing_after)
print(f"\n  Missing values after MICE: {missing_after} ✓")

# Reattach target
df_imputed["CombinedRisk_enc"] = df["CombinedRisk_enc"].values

# ─────────────────────────────────────────────
# 5. SMOTE
# ─────────────────────────────────────────────
print("\nSTEP 5 — SMOTE Oversampling")
print("=" * 60)

X = df_imputed.drop(columns=["CombinedRisk_enc"])
y = df_imputed["CombinedRisk_enc"]

print("  Class distribution BEFORE SMOTE:")
dist_before = pd.Series(y).value_counts().sort_index()
for cls, cnt in dist_before.items():
    print(f"    {le.inverse_transform([cls])[0]:12s} → {cnt}")
report["class_dist_before_smote"] = {str(le.inverse_transform([k])[0]): int(v) for k, v in dist_before.items()}

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

print("\n  Class distribution AFTER SMOTE:")
dist_after = pd.Series(y_res).value_counts().sort_index()
for cls, cnt in dist_after.items():
    print(f"    {le.inverse_transform([cls])[0]:12s} → {cnt}")
report["class_dist_after_smote"] = {str(le.inverse_transform([k])[0]): int(v) for k, v in dist_after.items()}
report["shape_after_smote"] = list(X_res.shape)

# ─────────────────────────────────────────────
# 6. MIN-MAX NORMALISATION
# ─────────────────────────────────────────────
print("\nSTEP 6 — Min-Max Normalisation")
print("=" * 60)

scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_res), columns=X_res.columns)
print("  All features scaled to [0, 1] ✓")
print(f"  Sample min/max check — Hb: [{X_scaled['Hb'].min():.3f}, {X_scaled['Hb'].max():.3f}]")

# ─────────────────────────────────────────────
# 7. RFE FEATURE SELECTION
# ─────────────────────────────────────────────
print("\nSTEP 7 — RFE Feature Selection")
print("=" * 60)

rf_estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rfe = RFE(estimator=rf_estimator, n_features_to_select=10, step=1)
rfe.fit(X_scaled, y_res)

selected_features = X_scaled.columns[rfe.support_].tolist()
eliminated_features = X_scaled.columns[~rfe.support_].tolist()

print(f"  Selected ({len(selected_features)}):")
for f in selected_features:
    print(f"    ✅ {f}")
print(f"\n  Eliminated ({len(eliminated_features)}):")
for f in eliminated_features:
    print(f"    ❌ {f}")

report["selected_features"] = selected_features
report["eliminated_features"] = eliminated_features

X_final = X_scaled[selected_features]

# ─────────────────────────────────────────────
# 8. TRAIN / VALIDATION / TEST SPLIT  80:10:10
# ─────────────────────────────────────────────
print("\nSTEP 8 — Train / Validation / Test Split (80:10:10)")
print("=" * 60)

X_train, X_temp, y_train, y_temp = train_test_split(
    X_final, y_res, test_size=0.20, random_state=42, stratify=y_res)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"  Train : {X_train.shape[0]} samples ({X_train.shape[0]/len(X_final)*100:.1f}%)")
print(f"  Val   : {X_val.shape[0]} samples ({X_val.shape[0]/len(X_final)*100:.1f}%)")
print(f"  Test  : {X_test.shape[0]} samples ({X_test.shape[0]/len(X_final)*100:.1f}%)")
report["split"] = {"train": int(X_train.shape[0]), "val": int(X_val.shape[0]), "test": int(X_test.shape[0])}

# ─────────────────────────────────────────────
# 9. SAVE OUTPUTS
# ─────────────────────────────────────────────
print("\nSTEP 9 — Saving Outputs")
print("=" * 60)

train_df = X_train.copy(); train_df["CombinedRisk_enc"] = y_train.values
val_df   = X_val.copy();   val_df["CombinedRisk_enc"]   = y_val.values
test_df  = X_test.copy();  test_df["CombinedRisk_enc"]  = y_test.values

train_df.to_csv(OUTPUT_DIR / "train_data.csv", index=False)
val_df.to_csv(OUTPUT_DIR / "val_data.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test_data.csv", index=False)

# Save label encoder mapping
with open(OUTPUT_DIR / "label_encoder_classes.json", "w") as f:
    json.dump(report["target_classes"], f, indent=2)

# Save report
with open(OUTPUT_DIR / "preprocessing_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("  ✅ train_data.csv")
print("  ✅ val_data.csv")
print("  ✅ test_data.csv")
print("  ✅ label_encoder_classes.json")
print("  ✅ preprocessing_report.json")

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)
print(f"  Final feature set : {selected_features}")
print(f"  Train samples     : {X_train.shape[0]}")
print(f"  Val samples       : {X_val.shape[0]}")
print(f"  Test samples      : {X_test.shape[0]}")
print(f"  Target classes    : {list(le.classes_)}")

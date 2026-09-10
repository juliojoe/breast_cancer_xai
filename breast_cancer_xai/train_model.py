"""
Train an explainable model for Breast Cancer diagnosis.
Dataset: Breast Cancer Wisconsin (Diagnostic) from scikit-learn (originally UCI / Kaggle).
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
import shap

# Load data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
# sklearn: 0 = malignant, 1 = benign. We invert so that 1 = Malignant (positive class)
y = 1 - data.target  # now 1 = Malignant, 0 = Benign

print("Class distribution (1=Malignant, 0=Benign):")
print(pd.Series(y).value_counts())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (helps some models, optional for RF but good practice)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model: Random Forest (excellent with TreeSHAP)
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print("\n=== Test Performance ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="roc_auc")
print(f"\n5-Fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# SHAP explainer (TreeExplainer is fast and exact for RF)
explainer = shap.TreeExplainer(model)

# Save artifacts
joblib.dump(model, "model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(explainer, "explainer.joblib")
joblib.dump(list(X.columns), "feature_names.joblib")
joblib.dump(
    {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
    },
    "data_split.joblib"
)

# Also save a sample background for SHAP
background = shap.sample(X_train_scaled, 100)
joblib.dump(background, "shap_background.joblib")

print("\nArtifacts saved successfully:")
print("  model.joblib, scaler.joblib, explainer.joblib,")
print("  feature_names.joblib, data_split.joblib, shap_background.joblib")

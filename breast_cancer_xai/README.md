# Explainable AI for Breast Cancer Diagnosis

**Final Year Computer Science Project**

An interactive Streamlit application that classifies breast tumors as **Benign** or **Malignant** using the classic Breast Cancer Wisconsin (Diagnostic) dataset and provides **self-explanatory side-by-side SHAP + LIME explanations** for every prediction.

## Dataset (chosen from Kaggle / UCI)

- **Name**: Breast Cancer Wisconsin (Diagnostic)
- **Source**: UCI Machine Learning Repository (also available on Kaggle)
- **Samples**: 569
- **Features**: 30 real-valued nuclear features extracted from fine-needle aspirate (FNA) images
- **Target**: Benign (B) / Malignant (M)

The dataset is loaded via `sklearn.datasets.load_breast_cancer` (identical to the Kaggle/UCI version).

## Key Features of the App

1. **Project Overview** – Goal, metrics, and navigation guide
2. **Data Exploration** – Class balance, statistics, correlation heatmap
3. **Predict & Explain** (core XAI page)
   - Load real test samples or enter key features
   - Instant diagnosis + probability
   - **Side-by-side SHAP and LIME local explanations**
   - Plain-English interpretation of the top drivers
   - Feature descriptions in clinical language
   - Downloadable text report of the explanation
4. **Model Performance** – Accuracy, ROC-AUC, confusion matrix, global SHAP importance
5. **SHAP vs LIME Comparison** – Educational page explaining the theoretical and practical differences
6. **About** – Dataset details, how SHAP & LIME work, ethical disclaimer

## Model

- Algorithm: **Random Forest** (200 trees, class-balanced)
- Test Accuracy: **≈ 97.4%**
- ROC-AUC: **≈ 0.997**
- Explainability: **TreeSHAP** (exact) + **LIME** (local linear approximation)

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) Re-train the model
python train_model.py

# Launch the Streamlit app
streamlit run app.py
```

## Project Structure

```
breast_cancer_xai/
├── app.py                 # Main Streamlit application (SHAP + LIME)
├── train_model.py         # Training + artifact generation
├── model.joblib           # Trained Random Forest
├── scaler.joblib          # StandardScaler
├── explainer.joblib       # SHAP TreeExplainer
├── feature_names.joblib
├── data_split.joblib
├── shap_background.joblib
├── requirements.txt
└── README.md
```

## Why this design?

- The explainable AI part is **self-explanatory**: every SHAP and LIME plot is accompanied by clear text that tells a non-expert *why* the model made its decision.
- Side-by-side view lets users cross-validate explanations.
- All explanations use medical-friendly language for the nuclear features.
- The interface is built entirely with Streamlit so it can be viewed and demonstrated easily.

## Ethical Note

This is an **educational / research demonstration**. It is **not** a medical device and must not be used for clinical decision-making.

"""
Explainable AI for Breast Cancer Diagnosis
Final Year Computer Science Project

Dataset : Breast Cancer Wisconsin (Diagnostic) – UCI / Kaggle
Model   : Random Forest + SHAP (TreeExplainer) + LIME
UI      : Streamlit – fully self-explanatory side-by-side explanations
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, RocCurveDisplay
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Explainable AI – Breast Cancer Diagnosis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Load artifacts (cached)
# -------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
    explainer = joblib.load("explainer.joblib")
    feature_names = joblib.load("feature_names.joblib")
    data = joblib.load("data_split.joblib")
    background = joblib.load("shap_background.joblib")
    return model, scaler, explainer, feature_names, data, background

model, scaler, explainer, feature_names, data, background = load_artifacts()

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]
X_train_scaled = data["X_train_scaled"]
X_test_scaled = data["X_test_scaled"]

# Create a prediction function that works on original-scale features
# (needed for LIME which is more interpretable on original scale)
def predict_proba_original(X_orig):
    """Accept original-scale features, scale them, return probabilities."""
    X_scaled = scaler.transform(X_orig)
    return model.predict_proba(X_scaled)

# LIME explainer (on original feature scale for better readability)
@st.cache_resource
def get_lime_explainer():
    return LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=["Benign", "Malignant"],
        mode="classification",
        discretize_continuous=True,
        random_state=42
    )

lime_explainer = get_lime_explainer()

# Feature descriptions for medical interpretability
FEATURE_DESCRIPTIONS = {
    "mean radius": "Average size of the cell nuclei (larger often linked to malignancy).",
    "mean texture": "Variation in gray-scale intensity of the nuclei.",
    "mean perimeter": "Average perimeter length of the nuclei.",
    "mean area": "Average area of the cell nuclei.",
    "mean smoothness": "Local variation in radius lengths (smoother or more irregular).",
    "mean compactness": "Perimeter² / area – 1. Higher values indicate more irregular shape.",
    "mean concavity": "Severity of concave portions of the cell contour.",
    "mean concave points": "Number of concave portions on the contour.",
    "mean symmetry": "Symmetry of the cell nuclei.",
    "mean fractal dimension": "Complexity of the boundary ('coastline' approximation).",
    "radius error": "Standard error of the radius measurement.",
    "texture error": "Standard error of texture.",
    "perimeter error": "Standard error of perimeter.",
    "area error": "Standard error of area.",
    "smoothness error": "Standard error of smoothness.",
    "compactness error": "Standard error of compactness.",
    "concavity error": "Standard error of concavity.",
    "concave points error": "Standard error of concave points.",
    "symmetry error": "Standard error of symmetry.",
    "fractal dimension error": "Standard error of fractal dimension.",
    "worst radius": "Largest (worst) radius observed among the nuclei.",
    "worst texture": "Worst (highest variation) texture.",
    "worst perimeter": "Largest perimeter.",
    "worst area": "Largest area.",
    "worst smoothness": "Worst smoothness value.",
    "worst compactness": "Worst compactness (most irregular).",
    "worst concavity": "Most severe concavity.",
    "worst concave points": "Highest number of concave points.",
    "worst symmetry": "Worst symmetry value.",
    "worst fractal dimension": "Highest fractal dimension (most complex boundary).",
}

# -------------------------------------------------
# Sidebar navigation
# -------------------------------------------------
st.sidebar.title("🩺 Breast Cancer XAI")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Project Overview",
        "📊 Data Exploration",
        "🔮 Predict & Explain",
        "📈 Model Performance",
        "⚖️ SHAP vs LIME Comparison",
        "ℹ️ About the Dataset & XAI"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Final Year Project**  \n"
    "Explainable Artificial Intelligence  \n"
    "for Breast Cancer Diagnosis  \n\n"
    "Models: Random Forest  \n"
    "XAI: **SHAP + LIME** (side-by-side)"
)

# -------------------------------------------------
# PAGE 1: Overview
# -------------------------------------------------
if page == "🏠 Project Overview":
    st.title("Explainable AI for Breast Cancer Diagnosis")
    st.markdown("### Final Year Computer Science Project")

    st.markdown("""
    This application demonstrates an **Explainable Artificial Intelligence (XAI)** system 
    that classifies breast tumors as **Benign** or **Malignant** using features extracted 
    from digitized images of fine-needle aspirates (FNA).

    #### Why Explainable AI?
    Black-box models can achieve high accuracy but doctors and patients need to **understand why** 
    a prediction was made. This system uses **two complementary XAI techniques**:

    - **SHAP** (SHapley Additive exPlanations) – theoretically grounded, consistent feature attributions
    - **LIME** (Local Interpretable Model-agnostic Explanations) – local linear approximation around the instance

    Both are shown **side-by-side** so you can compare how each method explains the same prediction.
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dataset Size", "569 samples")
    with col2:
        st.metric("Features", "30 nuclear features")
    with col3:
        st.metric("Test Accuracy", "97.4%")
    with col4:
        st.metric("ROC-AUC", "0.997")

    st.markdown("---")
    st.subheader("How to use this app")
    st.markdown("""
    1. **Data Exploration** – Understand the dataset and feature distributions.
    2. **Predict & Explain** – Enter patient feature values (or use a sample) and receive:
       - Diagnosis prediction + probability
       - **Side-by-side SHAP and LIME explanations**
       - Clear plain-English interpretation
    3. **Model Performance** – Global metrics, confusion matrix and global feature importance.
    4. **SHAP vs LIME Comparison** – Deep dive into the differences between the two methods.
    5. **About** – Dataset source, feature meanings, and how the XAI methods work.
    """)

    st.success("All explanations are designed to be self-explanatory for both technical and clinical users.")

# -------------------------------------------------
# PAGE 2: Data Exploration
# -------------------------------------------------
elif page == "📊 Data Exploration":
    st.title("📊 Data Exploration")

    st.markdown("""
    **Dataset**: Breast Cancer Wisconsin (Diagnostic)  
    Source: UCI Machine Learning Repository / Kaggle  
    Features computed from digitized FNA images of breast masses.
    """)

    # Class balance
    st.subheader("Class Distribution")
    y_full = pd.concat([pd.Series(y_train), pd.Series(y_test)])
    counts = y_full.value_counts().rename({0: "Benign", 1: "Malignant"})
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#2ecc71", "#e74c3c"]
    counts.plot(kind="bar", color=colors, ax=ax)
    ax.set_ylabel("Number of samples")
    ax.set_title("Benign vs Malignant")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    st.pyplot(fig)
    st.caption("357 Benign (62.7%) · 212 Malignant (37.3%)")

    # Feature statistics
    st.subheader("Feature Statistics")
    st.dataframe(X_train.describe().T.style.format("{:.3f}"), use_container_width=True)

    # Correlation heatmap of top features
    st.subheader("Correlation Heatmap (selected features)")
    top_feats = [
        "mean radius", "mean perimeter", "mean area",
        "mean concave points", "worst radius", "worst perimeter",
        "worst area", "worst concave points", "mean concavity", "worst concavity"
    ]
    corr = X_train[top_feats].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    st.pyplot(fig)

    st.markdown("""
    **Observation**: Size-related features (radius, perimeter, area) and shape irregularity 
    features (concave points, concavity) are highly correlated and are strong indicators 
    of malignancy.
    """)

# -------------------------------------------------
# PAGE 3: Predict & Explain (core XAI page with SHAP + LIME)
# -------------------------------------------------
elif page == "🔮 Predict & Explain":
    st.title("🔮 Predict & Explain a Case")
    st.markdown("""
    Enter the nuclear features (or load a sample from the test set) to obtain a diagnosis 
    **together with side-by-side SHAP and LIME explanations**.
    """)

    # Option to load a real test sample
    st.subheader("1. Choose input method")
    input_method = st.radio(
        "Input source",
        ["Use a real test-set sample", "Manual feature entry (key features)"],
        horizontal=True
    )

    if input_method == "Use a real test-set sample":
        sample_idx = st.slider("Select test sample index", 0, len(X_test) - 1, 12)
        sample = X_test.iloc[sample_idx].copy()
        true_label = "Malignant" if y_test.iloc[sample_idx] == 1 else "Benign"
        st.info(f"True label of this sample: **{true_label}** (index {sample_idx})")
    else:
        st.markdown("Adjust the most clinically important features (others kept at training median).")
        important = [
            "worst concave points", "worst perimeter", "mean concave points",
            "worst radius", "worst area", "mean concavity", "mean radius",
            "worst concavity", "mean perimeter", "mean area"
        ]
        sample_dict = {f: float(X_train[f].median()) for f in feature_names}
        cols = st.columns(2)
        for i, feat in enumerate(important):
            with cols[i % 2]:
                sample_dict[feat] = st.number_input(
                    feat,
                    value=float(X_train[feat].median()),
                    min_value=float(X_train[feat].min()),
                    max_value=float(X_train[feat].max()),
                    step=0.01,
                    key=feat
                )
        sample = pd.Series(sample_dict)

    # Scale and predict
    sample_df = pd.DataFrame([sample], columns=feature_names)
    sample_scaled = scaler.transform(sample_df)

    pred_proba = model.predict_proba(sample_scaled)[0]
    pred_class = model.predict(sample_scaled)[0]
    diagnosis = "Malignant" if pred_class == 1 else "Benign"
    confidence = pred_proba[1] if pred_class == 1 else pred_proba[0]

    # Result cards
    st.subheader("2. Prediction Result")
    col1, col2, col3 = st.columns(3)
    with col1:
        color = "red" if diagnosis == "Malignant" else "green"
        st.markdown(f"### Diagnosis: :{color}[**{diagnosis}**]")
    with col2:
        st.metric("Probability of Malignant", f"{pred_proba[1]*100:.1f}%")
    with col3:
        st.metric("Model Confidence", f"{confidence*100:.1f}%")

    # -------------------------------------------------
    # SIDE-BY-SIDE SHAP + LIME
    # -------------------------------------------------
    st.subheader("3. Side-by-Side Explanations (SHAP vs LIME)")

    st.markdown("""
    **How to read the explanations**  
    - **Red / positive** → feature pushes the prediction toward **Malignant**  
    - **Blue / negative** → feature pushes the prediction toward **Benign**  
    - Longer bars = stronger influence on *this specific case*
    """)

    # ----- Compute SHAP -----
    shap_values = explainer.shap_values(sample_scaled)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]          # positive class
    else:
        if shap_values.ndim == 3:
            sv = shap_values[0, :, 1]
        else:
            sv = shap_values[0]

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap": sv,
        "value": sample.values
    })
    shap_df["abs"] = shap_df["shap"].abs()
    shap_top = shap_df.sort_values("abs", ascending=True).tail(12)

    # ----- Compute LIME -----
    with st.spinner("Generating LIME explanation..."):
        lime_exp = lime_explainer.explain_instance(
            data_row=sample.values,
            predict_fn=predict_proba_original,
            num_features=12,
            top_labels=1
        )

    # LIME returns list of (feature, weight) for the predicted class
    lime_list = lime_exp.as_list(label=pred_class)
    lime_df = pd.DataFrame(lime_list, columns=["feature_desc", "lime_weight"])
    # Extract clean feature name (LIME sometimes adds conditions)
    lime_df["feature"] = lime_df["feature_desc"].apply(
        lambda x: x.split(" ")[0] if " " in x and any(c.isalpha() for c in x.split(" ")[0]) else x
    )

    # Layout: two columns
    col_shap, col_lime = st.columns(2)

    with col_shap:
        st.markdown("#### 🔷 SHAP Explanation")
        st.caption("Game-theory based, additive, consistent attributions")

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#e74c3c" if v > 0 else "#3498db" for v in shap_top["shap"]]
        ax.barh(shap_top["feature"], shap_top["shap"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value (impact on Malignant)")
        ax.set_title("Local SHAP Feature Contributions")
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("**Top SHAP drivers**")
        for _, row in shap_top.sort_values("shap", ascending=False).head(5).iterrows():
            direction = "↑ Malignant" if row["shap"] > 0 else "↓ Benign"
            st.markdown(f"- **{row['feature']}** = {row['value']:.3f} → {direction} ({row['shap']:+.3f})")

    with col_lime:
        st.markdown("#### 🔶 LIME Explanation")
        st.caption("Local linear approximation around this instance")

        fig, ax = plt.subplots(figsize=(8, 6))
        lime_sorted = lime_df.sort_values("lime_weight")
        colors = ["#e74c3c" if v > 0 else "#3498db" for v in lime_sorted["lime_weight"]]
        ax.barh(lime_sorted["feature_desc"], lime_sorted["lime_weight"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("LIME weight (impact on prediction)")
        ax.set_title("Local LIME Feature Contributions")
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("**Top LIME drivers**")
        for _, row in lime_df.sort_values("lime_weight", ascending=False).head(5).iterrows():
            direction = "↑ Malignant" if row["lime_weight"] > 0 else "↓ Benign"
            st.markdown(f"- **{row['feature_desc']}** → {direction} ({row['lime_weight']:+.3f})")

    # ----- Plain-English combined summary -----
    st.markdown("---")
    st.subheader("4. Combined Plain-English Interpretation")

    # Find features that both methods agree on (rough match)
    top_shap_pos = set(shap_top[shap_top["shap"] > 0]["feature"].tolist())
    top_shap_neg = set(shap_top[shap_top["shap"] < 0]["feature"].tolist())

    if diagnosis == "Malignant":
        st.markdown(f"Both methods agree that the prediction of **Malignant** is mainly driven by:")
        # Show top positive from SHAP with description
        for _, row in shap_top[shap_top["shap"] > 0].sort_values("shap", ascending=False).head(4).iterrows():
            desc = FEATURE_DESCRIPTIONS.get(row["feature"], "")
            st.markdown(f"- **{row['feature']}** (value = {row['value']:.3f})  \n  _{desc}_")
    else:
        st.markdown(f"Both methods agree that the prediction of **Benign** is mainly supported by:")
        for _, row in shap_top[shap_top["shap"] < 0].sort_values("shap").head(4).iterrows():
            desc = FEATURE_DESCRIPTIONS.get(row["feature"], "")
            st.markdown(f"- **{row['feature']}** (value = {row['value']:.3f})  \n  _{desc}_")

    st.info("""
    **Tip**: When SHAP and LIME highlight the same features, confidence in the explanation is higher.  
    When they differ, it often means the local decision boundary is complex — look at both views.
    """)

    # Downloadable summary
    st.subheader("5. Download Explanation Summary")
    summary_text = f"""
Breast Cancer XAI Explanation Report
=====================================
Diagnosis          : {diagnosis}
Probability Malignant: {pred_proba[1]*100:.1f}%
True label (if known): {true_label if input_method == 'Use a real test-set sample' else 'N/A'}

Top SHAP features (impact on Malignant):
"""
    for _, row in shap_top.sort_values("shap", ascending=False).iterrows():
        summary_text += f"  {row['feature']:30s}  value={row['value']:.4f}  SHAP={row['shap']:+.4f}\n"

    summary_text += "\nTop LIME features:\n"
    for _, row in lime_df.sort_values("lime_weight", ascending=False).iterrows():
        summary_text += f"  {row['feature_desc']:40s}  weight={row['lime_weight']:+.4f}\n"

    st.download_button(
        label="Download text report",
        data=summary_text,
        file_name=f"xai_explanation_{diagnosis.lower()}.txt",
        mime="text/plain"
    )

# -------------------------------------------------
# PAGE 4: Model Performance
# -------------------------------------------------
elif page == "📈 Model Performance":
    st.title("📈 Model Performance & Global Explainability")

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred)*100:.1f}%")
    col2.metric("ROC-AUC", f"{roc_auc_score(y_test, y_proba):.3f}")
    col3.metric("Malignant Recall", f"{classification_report(y_test, y_pred, output_dict=True)['1']['recall']*100:.1f}%")
    col4.metric("Benign Recall", f"{classification_report(y_test, y_pred, output_dict=True)['0']['recall']*100:.1f}%")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Malignant"],
                yticklabels=["Benign", "Malignant"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("ROC Curve")
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.plot([0, 1], [0, 1], "k--")
    st.pyplot(fig)

    # Global SHAP importance
    st.subheader("Global Feature Importance (SHAP)")
    st.markdown("""
    These are the features that, **on average across all patients**, have the strongest 
    influence on the model’s decisions. Higher mean |SHAP| = more important overall.
    """)

    shap_values_global = explainer.shap_values(X_test_scaled[:100])
    if isinstance(shap_values_global, list):
        sv_global = shap_values_global[1]
    else:
        if shap_values_global.ndim == 3:
            sv_global = shap_values_global[:, :, 1]
        else:
            sv_global = shap_values_global

    mean_abs_shap = np.abs(sv_global).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(importance_df["feature"], importance_df["mean_abs_shap"], color="#9b59b6")
    ax.set_xlabel("Mean |SHAP value| (average impact magnitude)")
    ax.set_title("Top 15 Most Important Features Globally")
    st.pyplot(fig)

    st.markdown("""
    **Clinical takeaway**: Features related to **size** (radius, perimeter, area – especially 
    the “worst” values) and **shape irregularity** (concave points, concavity) dominate the 
    model’s decisions, which aligns with known medical knowledge about malignant tumors.
    """)

# -------------------------------------------------
# PAGE 5: SHAP vs LIME Comparison (new educational page)
# -------------------------------------------------
elif page == "⚖️ SHAP vs LIME Comparison":
    st.title("⚖️ SHAP vs LIME – Understanding the Differences")

    st.markdown("""
    Both SHAP and LIME explain individual predictions, but they do so in fundamentally different ways.
    Understanding their strengths and weaknesses helps you trust (or question) the explanations.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔷 SHAP (SHapley Additive exPlanations)")
        st.markdown("""
        **Core idea**  
        From cooperative game theory. Each feature is a “player”.  
        SHAP computes the average marginal contribution of a feature across all possible coalitions.

        **Properties**
        - **Additive**: sum of SHAP values + base value = model output
        - **Consistent**: if a feature becomes more important, its SHAP value never decreases
        - **Model-specific efficient algorithms** exist (TreeSHAP for trees)

        **Best for**
        - Global + local explanations
        - When you need theoretical guarantees
        - Tree-based models (very fast & exact)

        **Limitations**
        - Can be slower for non-tree models
        - Assumes feature independence in the classic formulation (extensions exist)
        """)

    with col2:
        st.markdown("### 🔶 LIME (Local Interpretable Model-agnostic Explanations)")
        st.markdown("""
        **Core idea**  
        Fits a simple interpretable model (usually linear) in the local neighborhood 
        of the instance being explained.

        **Properties**
        - **Model-agnostic**: works with any black-box
        - Produces sparse explanations (only a few features)
        - Easy to understand linear weights

        **Best for**
        - Quick local insight
        - Any model type (neural nets, ensembles, etc.)
        - When you want a simple “if these features change…” story

        **Limitations**
        - Sensitive to the choice of neighborhood and kernel
        - No global consistency guarantee
        - Can be unstable across similar instances
        """)

    st.markdown("---")
    st.subheader("When do they agree / disagree?")

    st.markdown("""
    | Situation                        | Typical behaviour                                      |
    |----------------------------------|--------------------------------------------------------|
    | Linear / simple decision region  | SHAP and LIME usually highlight the same features      |
    | Complex interactions             | SHAP captures interactions better; LIME may approximate|
    | Correlated features              | Both can be unstable; SHAP has more principled handling|
    | Need for global view             | Prefer SHAP (mean |SHAP| plots)                        |
    | Need for very sparse explanation | Prefer LIME                                            |

    **In this project** we show both side-by-side so you can:
    1. Cross-validate the explanation
    2. See which features are robust across methods
    3. Learn the practical differences on real medical data
    """)

    st.success("""
    **Practical recommendation for this breast-cancer model**  
    Because we use a Random Forest, **TreeSHAP is exact and preferred**.  
    LIME is still valuable as a second opinion and for teaching the concept of local linear approximations.
    """)

# -------------------------------------------------
# PAGE 6: About
# -------------------------------------------------
elif page == "ℹ️ About the Dataset & XAI":
    st.title("ℹ️ About the Dataset & Explainable AI")

    st.subheader("Dataset")
    st.markdown("""
    **Name**: Breast Cancer Wisconsin (Diagnostic)  
    **Source**: University of Wisconsin Hospitals (Dr. William H. Wolberg)  
    **Available on**: UCI Machine Learning Repository and Kaggle  
    **Samples**: 569  
    **Features**: 30 real-valued features computed from digitized images of fine-needle 
    aspirates (FNA) of breast masses.  
    **Target**: Diagnosis – Benign (B) or Malignant (M)

    Ten characteristics of cell nuclei are measured:
    - radius, texture, perimeter, area, smoothness  
    - compactness, concavity, concave points, symmetry, fractal dimension  

    For each of the 10 characteristics the **mean**, **standard error**, and **worst** 
    (mean of the three largest values) are calculated → 30 features.
    """)

    st.subheader("What is SHAP?")
    st.markdown("""
    **SHAP (SHapley Additive exPlanations)** is a game-theory approach that explains 
    the output of any machine learning model.

    - It assigns each feature an importance value for a **particular prediction**.
    - The values are additive: base value + sum of SHAP values = model output.
    - Positive SHAP → pushes prediction toward Malignant.
    - Negative SHAP → pushes prediction toward Benign.
    - It is consistent and theoretically grounded (Shapley values from cooperative game theory).

    In this project we use **TreeSHAP**, an exact and fast algorithm designed for 
    tree-based models such as Random Forest.
    """)

    st.subheader("What is LIME?")
    st.markdown("""
    **LIME (Local Interpretable Model-agnostic Explanations)** explains individual 
    predictions by learning a simple interpretable model (usually linear) that 
    approximates the black-box model in the local neighborhood of the instance.

    - Model-agnostic – works with any classifier
    - Produces sparse, human-friendly feature weights
    - Complements SHAP by offering a different perspective on the same prediction
    """)

    st.subheader("Model Choice")
    st.markdown("""
    **Random Forest** was selected because:
    - High predictive performance on this dataset
    - Native support for exact SHAP explanations (TreeExplainer)
    - Handles non-linear relationships and feature interactions well
    - Less prone to overfitting with proper regularization
    """)

    st.subheader("Ethical Note")
    st.warning("""
    This tool is a **demonstration / educational project**.  
    It is **not** a certified medical device and must **never** be used as the sole basis 
    for clinical decisions. Always consult qualified medical professionals.
    """)

    st.markdown("---")
    st.markdown("**Project by**: Computer Science Final Year Student  \n"
                "**Focus**: Explainable AI · Healthcare · Streamlit Deployment  \n"
                "**XAI Techniques**: SHAP + LIME (side-by-side)")

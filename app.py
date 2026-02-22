import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


APP_TITLE = "Placebo Effect Prediction System"
MODEL_PATH_PRIMARY = "model.pkl"
MODEL_PATH_FALLBACK = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"
ENCODER_GENDER_PATH = "models/label_encoder_gender.pkl"
ENCODER_TREATMENT_PATH = "models/label_encoder_treatment.pkl"
DATASET_PATH = "data/dataset.csv"
X_SCALED_PATH = "data/X_scaled.csv"


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("For educational use only.")


@st.cache_data
def load_dataset():
    if os.path.exists(DATASET_PATH):
        return pd.read_csv(DATASET_PATH)
    return None


@st.cache_data
def load_scaled_features():
    if os.path.exists(X_SCALED_PATH):
        return pd.read_csv(X_SCALED_PATH)
    return None


@st.cache_resource
def load_model_and_preprocessors():
    model = None
    if os.path.exists(MODEL_PATH_PRIMARY):
        model = joblib.load(MODEL_PATH_PRIMARY)
    elif os.path.exists(MODEL_PATH_FALLBACK):
        model = joblib.load(MODEL_PATH_FALLBACK)
        joblib.dump(model, MODEL_PATH_PRIMARY)
    else:
        raise FileNotFoundError(
            "Model not found. Run training first to create models/best_model.pkl"
        )

    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    le_gender = joblib.load(ENCODER_GENDER_PATH) if os.path.exists(ENCODER_GENDER_PATH) else None
    le_treatment = joblib.load(ENCODER_TREATMENT_PATH) if os.path.exists(ENCODER_TREATMENT_PATH) else None

    return model, scaler, le_gender, le_treatment


def safe_encode(encoder, value, fallback=0):
    if encoder is None:
        return fallback
    try:
        return encoder.transform([value])[0]
    except Exception:
        return fallback


def get_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        importances = np.zeros(len(feature_names))
    return pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(
        "Importance", ascending=False
    )


def build_input_vector(
    dataset,
    age,
    gender,
    optimism,
    stress,
    anxiety,
    resilience,
    symptom_severity,
    le_gender,
    le_treatment,
):
    # Use dataset averages for features not collected from user
    defaults = dataset.mean(numeric_only=True) if dataset is not None else pd.Series()

    # Map symptom severity into stress to reflect symptom burden
    stress_level = float(np.clip((stress + symptom_severity) / 2.0, 1, 10))

    row = {
        "Age": age,
        "Openness": defaults.get("Openness", 6.5),
        "Conscientiousness": defaults.get("Conscientiousness", 6.5),
        "Extraversion": defaults.get("Extraversion", 6.0),
        "Agreeableness": defaults.get("Agreeableness", 6.5),
        "Neuroticism": defaults.get("Neuroticism", 5.0),
        "Optimism": optimism,
        "Stress_Level": stress_level,
        "Anxiety_Level": anxiety,
        "Emotional_Resilience": resilience,
        "Phase_Number": defaults.get("Phase_Number", 3),
        "Log_Enrollment": defaults.get("Log_Enrollment", 5.0),
        "Status_Completed": defaults.get("Status_Completed", 1),
        "Is_Placebo_Controlled": defaults.get("Is_Placebo_Controlled", 1),
        "Start_Year": defaults.get("Start_Year", 2010),
        "Start_Month": defaults.get("Start_Month", 6),
        "Condition_Encoded": defaults.get("Condition_Encoded", 0),
        "Gender_Encoded": safe_encode(le_gender, gender, fallback=0),
        "Treatment_Type_Encoded": safe_encode(le_treatment, "Sugar Pill", fallback=0),
    }
    return row


def prediction_block(model, scaler, feature_row):
    feature_df = pd.DataFrame([feature_row])
    if scaler is not None:
        X_scaled = scaler.transform(feature_df)
    else:
        X_scaled = feature_df.values
    proba = model.predict_proba(X_scaled)[0][1]
    pred = int(proba >= 0.5)
    return pred, proba, feature_df


dataset = load_dataset()
X_scaled = load_scaled_features()

try:
    model, scaler, le_gender, le_treatment = load_model_and_preprocessors()
except Exception as e:
    st.error(str(e))
    st.stop()


st.sidebar.header("Patient Input")
age = st.sidebar.slider("Age", 18, 80, 35)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
optimism = st.sidebar.slider("Optimism (1-10)", 1, 10, 7)
stress = st.sidebar.slider("Stress (1-10)", 1, 10, 4)
anxiety = st.sidebar.slider("Anxiety (1-10)", 1, 10, 4)
resilience = st.sidebar.slider("Emotional Resilience (1-10)", 1, 10, 7)
symptom_severity = st.sidebar.slider("Symptom Severity (1-10)", 1, 10, 5)


feature_row = build_input_vector(
    dataset,
    age,
    gender,
    optimism,
    stress,
    anxiety,
    resilience,
    symptom_severity,
    le_gender,
    le_treatment,
)

pred, proba, feature_df = prediction_block(model, scaler, feature_row)


col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Prediction")
    st.metric("Placebo Response Probability", f"{proba*100:.2f}%")

    if proba > 0.70:
        st.success("Recommendation: Placebo (Sugar Pill)")
        st.write("Dose: 1 tablet")
        st.write("Frequency: Twice daily")
        st.write("Duration: 7 days")
    else:
        st.warning("Recommendation: Generic Real Medication")
        st.write("Dose: 50 mg")
        st.write("Frequency: Once daily")
        st.write("Duration: 14 days")

with col2:
    st.subheader("Probability Gauge")
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2E86C1"},
                "steps": [
                    {"range": [0, 70], "color": "#F5B7B1"},
                    {"range": [70, 100], "color": "#A9DFBF"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "value": 70},
            },
            number={"suffix": "%"},
        )
    )
    st.plotly_chart(fig_gauge, use_container_width=True)


st.subheader("Feature Importance")
feature_names = list(feature_df.columns)
fi_df = get_feature_importance(model, feature_names)
fig_fi = px.bar(
    fi_df.head(12),
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top Feature Importances",
)
st.plotly_chart(fig_fi, use_container_width=True)


st.subheader("Patient Profile (Radar)")
radar_features = ["Optimism", "Stress_Level", "Anxiety_Level", "Emotional_Resilience"]
radar_values = [feature_row["Optimism"], feature_row["Stress_Level"], feature_row["Anxiety_Level"], feature_row["Emotional_Resilience"]]

fig_radar = go.Figure()
fig_radar.add_trace(
    go.Scatterpolar(
        r=radar_values,
        theta=radar_features,
        fill="toself",
        name="Patient",
    )
)
fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])))
st.plotly_chart(fig_radar, use_container_width=True)


st.subheader("Patient vs Dataset Average")
if dataset is not None:
    avg_vals = dataset[radar_features].mean(numeric_only=True).tolist()
    comp_df = pd.DataFrame(
        {
            "Feature": radar_features,
            "Patient": radar_values,
            "Dataset Average": avg_vals,
        }
    )
    fig_comp = px.bar(comp_df, x="Feature", y=["Patient", "Dataset Average"], barmode="group")
    st.plotly_chart(fig_comp, use_container_width=True)


st.subheader("Prediction Probability Distribution")
if X_scaled is not None and scaler is not None:
    try:
        probs = model.predict_proba(X_scaled.values)[:, 1]
        fig_hist = px.histogram(
            probs,
            nbins=30,
            title="Distribution of Predicted Probabilities (Dataset)",
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    except Exception:
        st.info("Probability distribution not available for this model.")


st.subheader("Correlation Heatmap")
if dataset is not None:
    corr_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
    corr = dataset[corr_cols].corr()
    fig_corr = px.imshow(corr, text_auto=False, aspect="auto", title="Correlation Heatmap")
    st.plotly_chart(fig_corr, use_container_width=True)


st.subheader("SHAP Explanation (Optional)")
with st.expander("Show SHAP explanation"):
    try:
        import shap

        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(feature_df)
            st.set_option("deprecation.showPyplotGlobalUse", False)
            shap.summary_plot(shap_values, feature_df, show=False)
            st.pyplot(bbox_inches="tight")
        else:
            st.info("SHAP is only shown for tree-based models in this demo.")
    except Exception as e:
        st.info(f"SHAP not available: {e}")


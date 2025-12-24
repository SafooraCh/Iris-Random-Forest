import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Iris Random Forest - Task 2", layout="wide")

@st.cache_resource
def load_bundle(path: str = "rf_iris_model.pkl"):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle

bundle = load_bundle()
model = bundle["model"]
feature_names = bundle["feature_names"]
target_names = bundle["target_names"]
metrics = bundle.get("metrics", {})

st.title("Task 2 — Iris Classification (Random Forest)")
st.write("Predict Iris species from feature inputs and show class probabilities.")

# Sidebar inputs
st.sidebar.header("Input Features")

defaults = {
    "sepal length (cm)": 5.4,
    "sepal width (cm)": 3.4,
    "petal length (cm)": 4.7,
    "petal width (cm)": 1.4,
}
ranges = {
    "sepal length (cm)": (4.0, 8.0),
    "sepal width (cm)": (2.0, 4.5),
    "petal length (cm)": (1.0, 7.0),
    "petal width (cm)": (0.1, 2.6),
}

values = []
for feat in feature_names:
    lo, hi = ranges.get(feat, (0.0, 10.0))
    dv = defaults.get(feat, (lo + hi) / 2)
    values.append(st.sidebar.slider(feat, float(lo), float(hi), float(dv), 0.1))

input_df = pd.DataFrame([values], columns=feature_names)

# Prediction
pred_class = int(model.predict(input_df)[0])
proba = model.predict_proba(input_df)[0]
pred_name = target_names[pred_class]
confidence = float(proba[pred_class])

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Your Input")
    st.dataframe(input_df, use_container_width=True)

    st.subheader("Model Metrics (from Kaggle)")
    if metrics:
        mdf = pd.DataFrame([metrics]).T.reset_index()
        mdf.columns = ["Metric", "Value"]
        st.dataframe(mdf, use_container_width=True, hide_index=True)
    else:
        st.info("Metrics not found inside the model file.")

with col2:
    st.subheader("Prediction")
    st.markdown(f"**Predicted class:** {pred_name}")
    st.markdown(f"**Confidence:** {confidence*100:.2f}%")

    proba_df = pd.DataFrame({"class": target_names, "probability": proba})
    fig = px.bar(proba_df, x="class", y="probability", title="Class Probabilities", range_y=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Task 2: Random Forest model trained in Kaggle and deployed using Streamlit.")

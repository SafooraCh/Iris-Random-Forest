import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
st.set_page_config(page_title="🌸 Iris RF Classifier", page_icon="🌸", layout="wide")
@st.cache_resource
def load_model():
    with open('rf_iris_model.pkl', 'rb') as f:  # Your downloaded file
        data = pickle.load(f)
    return data['model'], data['feature_names'], data['target_names']
model, feature_names, target_names = load_model()
st.title("🌸 **Iris Flower Classifier** - Random Forest")
st.markdown("***Trained on Kaggle | Metrics: Acc=97.8%, F1=97.8%***")
# Sidebar: Inputs
st.sidebar.header("📊 **Input Measurements (cm)**")
input_values = []
for feat in feature_names:
    if 'sepal length' in feat.lower(): default, minv, maxv = 5.4, 4.3, 7.9
    elif 'sepal width' in feat.lower(): default, minv, maxv = 3.4, 2.0, 4.4
    elif 'petal length' in feat.lower(): default, minv, maxv = 4.7, 1.0, 6.9
    else: default, minv, maxv = 1.4, 0.1, 2.5
    val = st.sidebar.slider(feat.title(), minv, maxv, default, 0.1)
    input_values.append(val)
input_df = pd.DataFrame([input_values], columns=feature_names)
# Columns Layout
col1, col2 = st.columns(2)
with col1:
    st.subheader("**📋 Your Input**")
    st.dataframe(input_df.T, use_container_width=True, hide_index=True)
with col2:
    st.subheader("**🎯 Prediction**")
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    conf = proba[pred] * 100
    
    st.balloons()  # Celebration!
    st.markdown(f"""
    <div style='text-align: center; padding: 20px; background: linear-gradient(45deg, #667eea 0%, #764ba2 100%); 
                color: white; border-radius: 15px; font-size: 24px;'>
        <strong>🌸 {target_names[pred].title()}</strong><br>
        <span style='font-size: 18px;'>Confidence: **{conf:.1f}%**</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Probability Chart
    prob_df = pd.DataFrame({'Species': target_names, 'Probability': proba})
    fig = px.bar(prob_df, x='Species', y='Probability', 
                 title='Prediction Probabilities', color='Probability', 
                 color_continuous_scale='viridis')
    st.plotly_chart(fig, use_container_width=True)
# Model Info
st.markdown("---")
col3, col4, col5 = st.columns(3)
col3.metric("🌳 Algorithm", "Random Forest")
col4.metric("🌿 Trees", "100")
col5.metric("📊 Classes", "3")
st.metric("⭐ Accuracy", "97.8%")
st.markdown("***Lab Task 2 - Deployed via GitHub & Streamlit Cloud***")

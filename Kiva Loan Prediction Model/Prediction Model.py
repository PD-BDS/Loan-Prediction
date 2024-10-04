import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import shap
from streamlit_shap import st_shap


st.title("Prediction using Supervised Machine Learning")

# Loading model and preprocessing objects
@st.cache_resource
def load_model_objects():
    model_xgb = joblib.load('model_xgb.joblib')
    scaler = joblib.load('scaler.joblib')
    ohe = joblib.load('ohe.joblib')
    return model_xgb, scaler, ohe

model_xgb, scaler, ohe = load_model_objects()

# SHAP explainer
explainer = shap.TreeExplainer(model_xgb)

# App description
with st.expander("What's this app?"):
    st.markdown("""
    This app is a KIVA loan amount predicter!
    """)

st.subheader('Describe your situation')

# User inputs
col1, col2 = st.columns(2)

with col1:
    sector = st.selectbox('sector', options=ohe.categories_[0])
    borrower_genders = st.selectbox('borrower_genders', options=ohe.categories_[1])
    country = st.selectbox('country', options=ohe.categories_[2])

with col2:
    term_in_months = st.number_input('term_in_months', min_value=0, max_value=100, value=1)
    lender_count = st.number_input('lender_count', min_value=0, max_value=100, value=1)


# Prediction button
if st.button('Predict Loan!'):
    # Prepare categorical features
    cat_features = pd.DataFrame({'sector': [sector], 'borrower_genders': [borrower_genders],'country': [country]})
    cat_encoded = pd.DataFrame(ohe.transform(cat_features).toarray(), columns=ohe.get_feature_names_out())

    # Prepare numerical features
    num_features = pd.DataFrame({
        'term_in_months': [term_in_months],
        'lender_count': [lender_count],
    })
    num_scaled = pd.DataFrame(scaler.transform(num_features), columns=num_features.columns)
    
    # Combine features
    features = pd.concat([num_scaled, cat_encoded], axis=1)
    
    # Make prediction
    predicted_price = model_xgb.predict(features)[0]
    
    # Display prediction
    st.metric(label="Predicted Loan:", value=f'{round(predicted_price)}')
    
    # Calculate and display price range
    lower_range = max(0, round(predicted_price - 110))
    upper_range = round(predicted_price + 110)
    st.write(f"Potential Variation: {lower_range} - {upper_range}")
    
    # SHAP explanation
    st.subheader('Price Factors Explained 🤖')
    shap_values = explainer.shap_values(features)
    st_shap(shap.force_plot(explainer.expected_value, shap_values, features), height=400, width=600)
    
    st.markdown("""
    This plot shows how each feature contributes to the predicted price:
    - Blue bars push the price lower
    - Red bars push the price higher
    - The length of each bar indicates the strength of the feature's impact
    """)

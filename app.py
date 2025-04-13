import streamlit as st
import numpy as np
import joblib


model = joblib.load("credit_fraud_model.pkl")
scaler = joblib.load("scaler.pkl")


v_features = [f'V{i}' for i in range(1, 29)]
other_features = ['Time', 'Amount']
feature_names = other_features + v_features  


st.title("💳 Credit Card Fraud Detection")
st.markdown("Enter transaction details below:")

user_input = []


st.subheader("Basic Transaction Details")
for feature in other_features:
    value = st.number_input(f"{feature}", min_value=0, step=1)
    user_input.append(value)


st.subheader("Anonymized Features (V1 to V28)")
for v in v_features:
    value = st.slider(f"{v}", min_value=-20.0, max_value=20.0, value=0.0, step=0.1)
    user_input.append(value)


if st.button("Detect Fraud"):
    input_array = np.array(user_input).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("⚠️ Fraud Detected!")
    else:
        st.success("✅ Transaction is Safe.")

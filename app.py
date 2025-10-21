import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import VotingClassifier

# ---------------------------------------------
# 🧩 Page Configuration
# ---------------------------------------------
st.set_page_config(
    page_title="Airline Passenger Satisfaction Predictor",
    page_icon="✈️",
    layout="wide"
)

st.title("SatisFly - Airline Passenger Satisfaction Prediction")
st.markdown("""
Predict whether a passenger is **Satisfied 😄** or **Neutral/Dissatisfied 😐**
based on flight experience features.
""")

# ---------------------------------------------
# ⚙️ Load Models and Scaler
# ---------------------------------------------
@st.cache_resource
def load_models():
    rf = joblib.load("rf_model.pkl")
    dt = joblib.load("dt_model.pkl")
    svm = joblib.load("svm_model.pkl")
    knn = joblib.load("knn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return rf, dt, svm, knn, scaler

rf_model, dt_model, svm_model, knn_model, scaler = load_models()

# Get expected feature names from scaler
expected_features = list(scaler.feature_names_in_)

# ---------------------------------------------
# ✍️ Sidebar Inputs
# ---------------------------------------------
st.sidebar.header("🧾 Input Passenger Details")

# Map sidebar inputs to features you actually used in training
inputs = {}

for feature in expected_features:
    if feature == "Customer Type":
        inputs[feature] = st.sidebar.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
        inputs[feature] = 1 if inputs[feature] == "Loyal Customer" else 0
    elif feature == "Type of Travel":
        inputs[feature] = st.sidebar.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
        inputs[feature] = 1 if inputs[feature] == "Business travel" else 0
    elif feature == "Class":
        inputs[feature] = st.sidebar.selectbox("Class", ["Business", "Eco", "Eco Plus"])
        inputs[feature] = 1 if inputs[feature] == "Business" else (0 if inputs[feature] == "Eco Plus" else 2)
    else:
        # For numeric features
        min_val = 0
        max_val = 5000
        default_val = 0
        if "Age" in feature:
            min_val, max_val, default_val = 7, 100, 30
        elif "Distance" in feature:
            min_val, max_val, default_val = 30, 7000, 500
        elif "Delay" in feature:
            min_val, max_val, default_val = 0, 2000, 0
        elif "Arrival" in feature:
            min_val, max_val, default_val = 0, 2000, 0
        inputs[feature] = st.sidebar.number_input(feature, min_val, max_val, default_val)

# Create dataframe and scale
input_df = pd.DataFrame([inputs])
scaled_input = scaler.transform(input_df)

# ---------------------------------------------
# 🔮 Predictions
# ---------------------------------------------
predictions = {
    'KNN': knn_model.predict(scaled_input)[0],
    'Random Forest': rf_model.predict(scaled_input)[0],
    'Decision Tree': dt_model.predict(scaled_input)[0],
    'SVM': svm_model.predict(scaled_input)[0]
}

# Soft voting
probs = {
    'KNN': knn_model.predict_proba(scaled_input)[0][1],
    'Random Forest': rf_model.predict_proba(scaled_input)[0][1],
    'Decision Tree': dt_model.predict_proba(scaled_input)[0][1],
    'SVM': svm_model.predict_proba(scaled_input)[0][1]
}

vote_prob = np.mean(list(probs.values()))
vote_pred = 1 if vote_prob >= 0.5 else 0

# ---------------------------------------------
# 📊 Display Results
# ---------------------------------------------
st.subheader("Prediction Result per Model")
st.table(pd.DataFrame.from_dict({**predictions, **{'Soft Voting Result': vote_pred}},
                                orient='index', columns=['Prediction']))

st.subheader("Prediction Probability per Model")
prob_df = pd.DataFrame.from_dict(probs, orient='index', columns=['Probability of Satisfied'])
st.bar_chart(prob_df)

st.subheader("Soft Voting Result")
if vote_pred == 1:
    st.success(f"✅ The passenger is **Satisfied!** 😄\nAverage Probability: {vote_prob:.2f}")
else:
    st.warning(f"⚠️ The passenger is **Neutral/Dissatisfied.** 😐\nAverage Probability: {vote_prob:.2f}")

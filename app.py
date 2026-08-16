# ==========================================
# app.py (Streamlit Application)
# Run via terminal: streamlit run app.py
# ==========================================
import streamlit as st
import pandas as pd
import joblib

# Set page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Load the saved model and feature list
@st.cache_resource
def load_assets():
    model = joblib.load('house_price_model.pkl')
    features = joblib.load('model_features.pkl')
    return model, features

model, model_features = load_assets()

st.title("🏠 House Price Prediction App")
st.markdown("Enter property details below to predict the estimated price.")

# Form layout for user inputs
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input("Area (sq ft)", min_value=500, max_value=25000, value=3500, step=100)
        bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3, step=1)
        bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=6, value=2, step=1)
        stories = st.selectbox("Number of Stories", options=[1, 2, 3, 4], index=1)
        parking = st.selectbox("Parking Spaces", options=[0, 1, 2, 3], index=1)

    with col2:
        mainroad = st.selectbox("Connected to Main Road?", options=["yes", "no"])
        guestroom = st.selectbox("Has Guestroom?", options=["yes", "no"])
        basement = st.selectbox("Has Basement?", options=["yes", "no"])
        hotwaterheating = st.selectbox("Has Hot Water Heating?", options=["yes", "no"])
        airconditioning = st.selectbox("Has Air Conditioning?", options=["yes", "no"])
        prefarea = st.selectbox("Located in Preferred Area?", options=["yes", "no"])
        furnishingstatus = st.selectbox("Furnishing Status", options=["furnished", "semi-furnished", "unfurnished"])

    submit_button = st.form_submit_button("Predict House Price")

if submit_button:
    # 1. Build a single-row DataFrame from the input
    raw_input = {
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'parking': parking,
        'mainroad': mainroad,
        'guestroom': guestroom,
        'basement': basement,
        'hotwaterheating': hotwaterheating,
        'airconditioning': airconditioning,
        'prefarea': prefarea,
        'furnishingstatus': furnishingstatus
    }
    input_df = pd.DataFrame([raw_input])

    # 2. One-hot encode using get_dummies
    encoded_input = pd.get_dummies(input_df, drop_first=True)

    # 3. Align input columns with model training features
    aligned_input = pd.DataFrame(0, index=[0], columns=model_features)
    for col in encoded_input.columns:
        if col in aligned_input.columns:
            aligned_input[col] = encoded_input[col].values

    # 4. Predict and display results
    prediction = model.predict(aligned_input)[0]
    
    st.success(f"### 🏷️ Estimated Price: Rs {prediction:,.2f}")
import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")

st.title("🏡 California House Price Predictor")

st.markdown("Use the sliders below to input house information and get a predicted price.")

# Input form
MedInc = st.slider("Median Income", 0.0, 20.0, 5.0)
HouseAge = st.slider("House Age", 1, 60, 20)
AveRooms = st.slider("Average Rooms", 1.0, 15.0, 5.0)
AveBedrms = st.slider("Average Bedrooms", 0.5, 5.0, 1.0)
Population = st.slider("Population", 100, 5000, 1000)
AveOccup = st.slider("Average Occupants per Household", 0.5, 10.0, 3.0)
Latitude = st.slider("Latitude", 32.0, 42.0, 36.0)
Longitude = st.slider("Longitude", -125.0, -114.0, -120.0)

features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])

# Predict
if st.button("Predict House Price"):
    prediction = model.predict(features)[0]
    st.success(f"🏷️ Predicted House Value: ${prediction * 100000:.2f}")
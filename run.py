import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Crop Production Prediction", layout="centered")

st.title("🌾 Crop Production Prediction App")
st.subheader("Predict crop production based on environmental and agricultural inputs")

st.write("""
This app demonstrates a machine learning model that predicts crop production using key factors such as area, rainfall, temperature, humidity, soil type, irrigation method, season, and crop type.
""")

st.markdown("## 🔧 Input Parameters")

def parse_input(label, slider_val, text_val, min_val, max_val):
    """Try to parse user text input, fallback to slider, and show a warning on failure."""
    try:
        val = float(text_val)
        if min_val <= val <= max_val:
            return val, None
        else:
            return slider_val, f"❌ {label} must be between {min_val} and {max_val}."
    except ValueError:
        return slider_val, f"❌ {label} must be a number."

st.write("Adjust the sliders or enter values manually:")

col1, col2 = st.columns(2)

with col1:
    area_slider = st.slider("📏 Area (hectares)", 0.1, 1000.0, 1.0, step=0.1)
    area_text = st.text_input("✏️ Or enter area manually", value=str(area_slider))
    area, area_error = parse_input("Area", area_slider, area_text, 0.1, 1000.0)
    if area_error:
        st.warning(area_error)

    temp_slider = st.slider("🌡️ Temperature (°C)", 0.0, 45.0, 25.0, step=0.1)
    temp_text = st.text_input("✏️ Or enter temperature manually", value=str(temp_slider))
    temperature, temp_error = parse_input("Temperature", temp_slider, temp_text, 0.0, 55.0)
    if temp_error:
        st.warning(temp_error)

with col2:
    rain_slider = st.slider("🌧️ Rainfall (mm)", 0.0, 3500.0, 100.0, step=1.0)
    rain_text = st.text_input("✏️ Or enter rainfall manually", value=str(rain_slider))
    rainfall, rain_error = parse_input("Rainfall", rain_slider, rain_text, 0.0, 3500.0)
    if rain_error:
        st.warning(rain_error)

    humid_slider = st.slider("💧 Humidity (%)", 0.0, 100.0, 60.0, step=0.1)
    humid_text = st.text_input("✏️ Or enter humidity manually", value=str(humid_slider))
    humidity, humid_error = parse_input("Humidity", humid_slider, humid_text, 0.0, 100.0)
    if humid_error:
        st.warning(humid_error)

# --- Categorical Inputs ---
st.markdown("---")
st.markdown("## 🧠 Categorical Inputs")

soil_type = st.selectbox(
    "⛰️ Soil Type",
    ["Alluvial", "Arid and Desert", "Black", "Black cotton", "Clay", "Clay loam", "Drained loam", "Dry Sandy",
     "Gravelly Sand", "Heavy clay", "Heavy Cotton", "Laterite", "Light Sandy", "Loam", "Loamy Sand",
     "Medium Textured", "Medium Textured Clay", "Red", "Red laterite", "River basins", "Sandy",
     "Sandy loam", "Teelah", "Well drained"]
)

irrigation_method = st.selectbox(
    "🚿 Irrigation Method",
    ["Drip", "Spray", "Basin"]
)

season = st.selectbox(
    "📆 Season",
    ["Kharif", "Rabi", "Zaid"]
)

crop_type = st.selectbox(
    "🌱 Crop Type",
    ["Arecanut", "Blackgram", "Cardamum", "Cashew", "Cocoa", "Coconut",
     "Coffee", "Ginger", "Groundnut", "Paddy", "Tea"]
)

# --- Load encoders and model ---
@st.cache_resource
def load_resources():
    soil_le = joblib.load('Soil type_le.pkl')
    irrigation_le = joblib.load('Irrigation_le.pkl')
    crop_le = joblib.load('Crops_le.pkl')
    season_le = joblib.load('season_le.pkl')
    model = joblib.load('best_model.pkl')  # Must be trained with this exact input order
    return soil_le, irrigation_le, crop_le, season_le, model

def encode_category(le, category):
    try:
        return le.transform([category.lower()])[0] + 1
    except ValueError:
        return 0  # fallback for unknown category

# --- Prediction ---
if st.button("🔍 Predict Crop Production"):
    if temp_error or rain_error or humid_error or area_error:
        st.error("⚠️ Please correct the invalid inputs before predicting.")
    else:
        # Load model and encoders
        soil_le, irrigation_le, crop_le, season_le, model = load_resources()

        # Encode categories
        soil_encoded = encode_category(soil_le, soil_type)
        irrigation_encoded = encode_category(irrigation_le, irrigation_method)
        crop_encoded = encode_category(crop_le, crop_type)
        season_encoded = encode_category(season_le, season)

        # Prepare input features in correct order
        features = np.array([[area, rainfall, temperature, humidity, soil_encoded, irrigation_encoded, season_encoded, crop_encoded]])

        # Predict
        predicted_yield = model.predict(features)[0]

        # Display result
        st.success("✅ Prediction complete!")
        st.markdown("## 📊 Prediction Summary")
        st.write(f"**Predicted Yield:** {predicted_yield:.2f} kg/hectare")

        st.markdown("### 📋 Input Recap")
        st.write(f"- **Area:** {area} hectares")
        st.write(f"- **Rainfall:** {rainfall} mm")
        st.write(f"- **Temperature:** {temperature} °C")
        st.write(f"- **Humidity:** {humidity} %")
        st.write(f"- **Soil Type:** {soil_type}")
        st.write(f"- **Irrigation Method:** {irrigation_method}")
        st.write(f"- **Season:** {season}")
        st.write(f"- **Crop Type:** {crop_type}")

else:
    st.info("ℹ️ Adjust the inputs and click **Predict** to see the predicted crop production.")

st.markdown("---")
st.caption("Created by ML Vivala Five! All rights reserved. © 2025")

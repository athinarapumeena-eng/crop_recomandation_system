import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Crop Recommendation System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main { padding: 0rem 1rem; }
.metric-box { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_and_train_model():

    # Load dataset
    df = pd.read_csv("Crop_recommendation.csv")

    # ✅ CLEAN COLUMN NAMES (IMPORTANT FIX)
    df.columns = df.columns.str.strip().str.lower()

    # Normalize crop names
    df['label'] = df['label'].str.capitalize()

    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    accuracy = model.score(X_test_scaled, y_test)

    return model, scaler, df, accuracy


# ---------------- CROP INFO ----------------
def get_crop_info(crop):

    crop_info = {
        "Rice": {
            "season": "Monsoon/Autumn",
            "ideal_temp": "20-25°C",
            "water_need": "High (150-250 mm)",
            "soil_type": "Clay/Loamy",
            "benefits": "Rich in carbohydrates, staple food",
            "tips": "Requires puddled fields and water management"
        },
        "Wheat": {
            "season": "Winter/Spring",
            "ideal_temp": "15-25°C",
            "water_need": "Moderate",
            "soil_type": "Loamy",
            "benefits": "Protein-rich crop",
            "tips": "Best for crop rotation"
        },
        "Maize": {
            "season": "Summer/Monsoon",
            "ideal_temp": "21-27°C",
            "water_need": "Moderate",
            "soil_type": "Well-drained loam",
            "benefits": "Animal feed & industrial use",
            "tips": "Needs proper spacing"
        }
    }

    return crop_info.get(crop, {})


# ---------------- APP UI ----------------
st.title("🌱 Smart Crop Recommendation System")
st.markdown("**Intelligent agricultural guidance using Machine Learning**")

model, scaler, training_data, accuracy = load_and_train_model()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("📊 System Info")
    st.metric("Model Accuracy", f"{accuracy:.2%}")
    st.metric("Training Samples", len(training_data))

    # ✅ FIXED LINE (was 'crop')
    st.metric("Crops Available", training_data['label'].nunique())

    st.divider()

    st.caption("Nitrogen (N): Plant growth promoter")
    st.caption("Phosphorus (P): Root development")
    st.caption("Potassium (K): Stress resistance")
    st.caption("Temperature: Climate")
    st.caption("Humidity: Moisture level")
    st.caption("pH: Soil acidity")
    st.caption("Rainfall: Annual precipitation")


# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🌾 Enter Soil & Climate Parameters")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        N = st.number_input("Nitrogen", 0, 150, 50)
        temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
        ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)

    with col_b:
        P = st.number_input("Phosphorus", 0, 150, 40)
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 70.0)

    with col_c:
        K = st.number_input("Potassium", 0, 150, 40)
        rainfall = st.number_input("Rainfall (mm)", 0.0, 300.0, 150.0)

    st.divider()

    # ---------------- PREDICTION ----------------
    if st.button("🎯 Get Recommendation", use_container_width=True):

        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]
        confidence = np.max(model.predict_proba(input_scaled)) * 100

        st.success(f"✅ Recommended Crop: **{prediction}**")
        st.metric("Confidence", f"{confidence:.1f}%")

        crop_details = get_crop_info(prediction)

        if crop_details:
            st.info(f"💡 Tip: {crop_details['tips']}")

        st.balloons()


# ---------------- SUMMARY TABLE ----------------
with col2:
    st.subheader("📈 Feature Summary")

    summary_df = pd.DataFrame({
        "Parameter": ["N", "P", "K", "Temperature", "Humidity", "pH", "Rainfall"],
        "Value": [N, P, K, temperature, humidity, ph, rainfall]
    })

    st.dataframe(summary_df, use_container_width=True)


st.divider()
st.markdown(
"""
<div style='text-align:center'>
🚀 <b>Smart Crop Recommendation System</b><br>
<small>Powered by Machine Learning | Streamlit</small>
</div>
""",
unsafe_allow_html=True
)
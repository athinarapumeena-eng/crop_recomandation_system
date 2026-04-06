import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os
from datetime import datetime

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

@st.cache_resource
def load_and_train_model():
    """Load or train the crop recommendation model"""
    
    # Load data from CSV file
    df = pd.read_csv("Crop_recommendation.csv")
    
    # Normalize crop names to title case for consistency with crop_info dictionary
    df['label'] = df['label'].str.capitalize()
    
    X = df.drop("label", axis=1)
    y = df["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X_train_scaled, y_train)
    
    accuracy = model.score(X_test_scaled, y_test)
    
    return model, scaler, df, accuracy

def get_crop_info(crop):
    """Return information about recommended crop"""
    crop_info = {
        "Rice": {
            "season": "Monsoon/Autumn",
            "ideal_temp": "20-25°C",
            "water_need": "High (150-250 mm)",
            "soil_type": "Clay/Loamy",
            "benefits": "Rich in carbohydrates, staple food",
            "tips": "Requires puddled fields and proper water management"
        },
        "Wheat": {
            "season": "Winter/Spring",
            "ideal_temp": "15-25°C",
            "water_need": "Moderate (40-50 cm)",
            "soil_type": "Loamy/Well-drained",
            "benefits": "Protein-rich, long shelf-life",
            "tips": "Sow in winter, harvest in spring. Excellent for rotation"
        },
        "Maize": {
            "season": "Summer/Monsoon",
            "ideal_temp": "21-27°C",
            "water_need": "Moderate-High (60 cm)",
            "soil_type": "Well-drained loam",
            "benefits": "Versatile, animal feed, industrial use",
            "tips": "Requires spacing and timely irrigation during grain filling"
        },
        "Cotton": {
            "season": "Spring/Summer",
            "ideal_temp": "20-30°C",
            "water_need": "High (6-8 irrigations)",
            "soil_type": "Black soil/Deep loam",
            "benefits": "Cash crop, textile industry",
            "tips": "Requires deep soil and excellent drainage"
        },
        "Sugarcane": {
            "season": "Year-round",
            "ideal_temp": "21-27°C",
            "water_need": "Very High (150-250 cm)",
            "soil_type": "Loamy/Alluvial",
            "benefits": "Sugar production, high returns",
            "tips": "Requires 12-18 months growth period"
        },
        "Millet": {
            "season": "Summer/Monsoon",
            "ideal_temp": "25-35°C",
            "water_need": "Low (40-50 cm)",
            "soil_type": "Light sandy soil",
            "benefits": "Drought-resistant, nutritious",
            "tips": "Ideal for arid/semi-arid regions"
        },
        "Pulses": {
            "season": "Winter/Summer",
            "ideal_temp": "20-30°C",
            "water_need": "Low-Moderate (40-60 cm)",
            "soil_type": "Well-drained",
            "benefits": "High protein, nitrogen-fixing",
            "tips": "Excellent for crop rotation"
        },
        "Groundnut": {
            "season": "Summer/Monsoon",
            "ideal_temp": "24-28°C",
            "water_need": "Moderate (60-90 cm)",
            "soil_type": "Light sandy loam",
            "benefits": "Oil content, protein-rich",
            "tips": "Requires well-drained sandy soil"
        },
        "Soybean": {
            "season": "Monsoon",
            "ideal_temp": "20-30°C",
            "water_need": "Moderate (60-70 cm)",
            "soil_type": "Well-drained loam",
            "benefits": "High protein, oil content",
            "tips": "Modern crop with market demand"
        }
    }
    return crop_info.get(crop, {})


st.title("🌱 Smart Crop Recommendation System")
st.markdown("**Intelligent agricultural guidance using Machine Learning**")

model, scaler, training_data, accuracy = load_and_train_model()

with st.sidebar:
    st.header("📊 System Info")
    st.metric("Model Accuracy", f"{accuracy:.2%}")
    st.metric("Training Samples", len(training_data))
    st.metric("Crops Available", training_data['crop'].nunique())
    
    st.divider()
    st.subheader("📖 Feature Guide")
    st.caption("**Nitrogen (N):** Plant growth promoter (0-150)")
    st.caption("**Phosphorus (P):** Root development (0-150)")
    st.caption("**Potassium (K):** Stress resistance (0-150)")
    st.caption("**Temperature:** Climate (0-50°C)")
    st.caption("**Humidity:** Moisture level (0-100%)")
    st.caption("**pH:** Soil acidity/alkalinity (0-14)")
    st.caption("**Rainfall:** Annual precipitation (0-300 mm)")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🌾 Enter Soil & Climate Parameters")
    
    col_a, col_b, col_c = st.columns(3, gap="medium")
    
    with col_a:
        N = st.number_input("Nitrogen (N)", min_value=0, max_value=150, value=50, step=5)
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.5)
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    
    with col_b:
        P = st.number_input("Phosphorus (P)", min_value=0, max_value=150, value=40, step=5)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
    
    with col_c:
        K = st.number_input("Potassium (K)", min_value=0, max_value=150, value=40, step=5)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=150.0, step=5.0)
    
    st.divider()
    
    # Recommendation button
    if st.button("🎯 Get Recommendation", use_container_width=True, type="primary"):
        try:
            # Prepare input
            input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            input_scaled = scaler.transform(input_data)
            
            # Predict
            prediction = model.predict(input_scaled)[0]
            confidence = np.max(model.predict_proba(input_scaled)) * 100
            
            # Display result
            st.success(f"### ✅ Recommended Crop: **{prediction}**", icon="✅")
            st.metric("Confidence Level", f"{confidence:.1f}%")
            
            # Show crop details
            crop_details = get_crop_info(prediction)
            
            if crop_details:
                st.markdown("#### 📋 Crop Details:")
                
                detail_cols = st.columns(2)
                
                with detail_cols[0]:
                    st.markdown(f"**🌡️ Ideal Temperature:** {crop_details['ideal_temp']}")
                    st.markdown(f"**💧 Water Need:** {crop_details['water_need']}")
                    st.markdown(f"**🪨 Soil Type:** {crop_details['soil_type']}")
                
                with detail_cols[1]:
                    st.markdown(f"**📅 Season:** {crop_details['season']}")
                    st.markdown(f"**✨ Benefits:** {crop_details['benefits']}")
                
                st.info(f"**💡 Tip:** {crop_details['tips']}")
            
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Error in prediction: {str(e)}")

with col2:
    st.subheader("📈 Feature Distribution")
    
    feature_summary = {
        "Parameter": ["Nitrogen", "Phosphorus", "Potassium", "Temperature", "Humidity", "pH", "Rainfall"],
        "Value": [N, P, K, temperature, humidity, ph, rainfall],
        "Unit": ["N", "P", "K", "°C", "%", "scale", "mm"]
    }
    
    summary_df = pd.DataFrame(feature_summary)
    st.dataframe(summary_df, width='stretch', hide_index=True)


st.divider()
st.markdown("""
    <div style='text-align: center'>
    <p>🚀 <strong>Smart Crop Recommendation System v2.0</strong></p>
    <p style='font-size: 0.8rem; color: gray;'>Powered by Machine Learning | Built with Streamlit</p>
    </div>
""", unsafe_allow_html=True)
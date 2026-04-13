import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import model_engine
import weather_service
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------
# 🎨 UI CONFIG
# -------------------------------
st.set_page_config(page_title="🌾 Sugarcane Red Rot Forecast", layout="wide")

st.title("🌾 Sugarcane Red Rot Outbreak Monitor")
st.markdown("""
Predicting fungal outbreak risks for Sugarcane using **Real-Time Weather Data** (Open-Meteo) 
and **XGBoost ML Architecture**.
""")

# -------------------------------
# 🧠 MODEL SETUP
# -------------------------------
model = model_engine.load_model()
if model is None:
    st.warning("⚠️ Model not found. Initializing training on historical data...")
    with st.spinner("Training research-grade model, please wait..."):
        csv_path = os.path.join(BASE_DIR, "POWER_Point_Daily_20160101_20251231_016d70N_074d69E_LST.csv")
        model = model_engine.train_and_save_model(csv_path)
    st.success("✅ Model trained and ready!")

# -------------------------------
# 🌍 LIVE DATA
# -------------------------------
st.sidebar.header("📍 Monitoring Location")
lat = st.sidebar.number_input("Latitude", value=16.70)
lon = st.sidebar.number_input("Longitude", value=74.69)

# 🔥 Upgrade: Regional Warning (Level 5)
st.sidebar.warning("⚠️ Model calibrated for Maharashtra sugarcane regions. Soil and variety factors in other states may vary.")

st.sidebar.info("Current Location: Sangli, Maharashtra (Sugarcane Hub)")

with st.spinner("Fetching live 7-day forecast (Continuous Lookback)..."):
    forecast_df = weather_service.fetch_weather_forecast(lat, lon)

if forecast_df is not None:
    # 1. Feature Engineering on LIVE data (With 7-day context)
    processed_df = model_engine.engineer_features(forecast_df)
    
    # 2. Predict Risk & Confidence (XGBoost Probabilities)
    X_live = processed_df[model_engine.FEATURES]
    preds = model.predict(X_live)
    probs = model.predict_proba(X_live)
    
    processed_df['predicted_risk'] = preds
    processed_df['confidence'] = [max(p) for p in probs]
    
    # Get Current Risk (Latest)
    latest_row = processed_df.iloc[-1]
    current_risk_val = int(latest_row['predicted_risk'])
    current_conf = latest_row['confidence']
    
    risk_labels = {0: "🟢 Low", 1: "🟡 Medium", 2: "🔴 High"}
    risk_colors = {0: "green", 1: "orange", 2: "red"}
    
    # -------------------------------
    # 📊 DASHBOARD LAYOUT
    # -------------------------------
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Current Risk Level")
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = current_risk_val,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Level: {risk_labels[current_risk_val]}", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [0, 2], 'tickvals': [0, 1, 2], 'ticktext': ["Low", "Med", "High"]},
                'bar': {'color': risk_colors[current_risk_val]},
                'steps': [
                    {'range': [0, 0.7], 'color': 'lightgreen'},
                    {'range': [0.7, 1.4], 'color': 'wheat'},
                    {'range': [1.4, 2], 'color': 'salmon'}]} ))
        st.plotly_chart(fig, use_container_width=True)
        
        # 🔥 Upgrade: Confidence Meter
        st.write(f"**Model Confidence:** {float(current_conf):.1%}")
        st.progress(float(current_conf))
        
        # 🔥 Upgrade: Context-Aware Advisory (Level 6: XAI)
        st.subheader("🛠️ Farm Advisory & Analysis")
        
        # 🧠 Explainable AI: Reasoning Logic
        reasons = []
        if latest_row['humidity'] > 85: reasons.append(f"High peak humidity ({latest_row['humidity']:.1f}%) favoring spore germination.")
        if latest_row['humidity_streak'] > 2: reasons.append(f"Extended humidity streak ({latest_row['humidity_streak']:.0f} hours) ensures fungal survival.")
        if latest_row['wet_spell'] > 3: reasons.append(f"Continuous wet spell ({latest_row['wet_spell']:.0f} days) provides travel medium for spores.")
        if latest_row['rain_3d_sum'] > 20: reasons.append(f"Significant 3-day rainfall ({latest_row['rain_3d_sum']:.1f}mm) promotes soil-borne spread.")
        if latest_row['temp_3d_avg'] >= 25 and latest_row['temp_3d_avg'] <= 30: reasons.append(f"Optimal thermal window ({latest_row['temp_3d_avg']:.1f}°C) for Red Rot growth.")
        if latest_row['growth_stage'] == 3: reasons.append("Crop in 'Grand Growth' phase: height and canopy density increase susceptibility.")
        
        if reasons:
            st.info("💡 **Environmental Risk Drivers:**\n- " + "\n- ".join(reasons))
        else:
            st.info("💡 **Key Drivers:** Environmental conditions are currently stable and outside the optimal fungal growth window.")

        if current_risk_val == 2:
            st.error("🚨 **CRITICAL RISK**: Outbreak conditions high. Avoid heavy irrigation. Prepare fungicide (e.g. Carbendazim). Check lower leaves for orange/red spots.")
        elif current_risk_val == 1:
            st.warning("⚠️ **WARNING**: Favorable conditions detected. Monitor fields for 'dead hearts' in tillers. Ensure proper drainage.")
        else:
            st.success("✅ **STABLE**: Low risk. Maintain regular field sanitation and monitor for stem borers.")

    with col2:
        st.subheader("📅 7-Day Forecast Trend")
        fig_trend = px.line(processed_df.tail(7), x='date', y='predicted_risk', 
                            title="Forecasted Outbreak Probability Evolution", 
                            labels={'predicted_risk': 'Risk Level', 'date': 'Forecast Date'},
                            markers=True)
        fig_trend.update_yaxes(range=[-0.5, 2.5], tickvals=[0, 1, 2], ticktext=["Low", "Medium", "High"])
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # 🔥 Upgrade: Full Feature Transparency (XAI)
        st.subheader("📋 Full Feature Transparency (Model Input Context)")
        st.markdown("Raw biological features fed into the XGBoost model for the upcoming 7 days.")
        st.dataframe(processed_df.tail(7))

    # 📈 BIOLOGICAL DRIVERS
    st.divider()
    st.subheader("🔬 Biological Context")
    cols = st.columns(4)
    cols[0].metric("Humidity Max", f"{latest_row['humidity']:.1f} %", delta="Peak Window")
    cols[1].metric("Wet Spell", f"{latest_row['wet_spell']:.0f} Days")
    cols[2].metric("Growth Stage", f"{latest_row['growth_stage']:.0f}", help="1: Germ, 2: Till, 3: Grand, 4: Mat")
    cols[3].metric("Temp Avg (3d)", f"{latest_row['temp_3d_avg']:.1f} °C")

else:
    st.error("Failed to load forecast data. Please verify your internet connection.")

from fastapi import FastAPI, Depends, HTTPException, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import os
import shutil
import uvicorn
import subprocess
import sys
from datetime import datetime

from db.database import Base, engine, get_db
from db import models, schemas
from services.weather_service import get_current_weather
from services.ml_service import predict_crop_disease, get_supported_crops
from services.llm_service import extract_ml_features, generate_recommendation, generate_unsupported_crop_recommendation

# Create db tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Crop Disease Outbreak Prediction System",
    description="Multi-Modal Agentic Backend System for predicting crop diseases and offering LLM recommendations."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directories
static_dir = os.path.join(os.path.dirname(__file__), "static")
uploads_dir = os.path.join(static_dir, "uploads")
for d in [static_dir, uploads_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Risk Monitor Process Manager
streamlit_process = None

@app.on_event("startup")
async def startup_event():
    global streamlit_process
    # Launch Streamlit in the background: streamlit run data/app.py --server.port 8501 --server.address 0.0.0.0
    # We use sys.executable -m streamlit to ensures it uses the same environment
    cmd = [sys.executable, "-m", "streamlit", "run", "data/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
    try:
        streamlit_process = subprocess.Popen(cmd)
        print("🚀 Sugarcane Risk Monitor (Streamlit) started on port 8501")
    except Exception as e:
        print(f"❌ Failed to start Streamlit: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    global streamlit_process
    if streamlit_process:
        print("🛑 Shutting down Sugarcane Risk Monitor...")
        streamlit_process.terminate()
        try:
            streamlit_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            streamlit_process.kill()

@app.get("/")
async def root():
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.post("/predict")
async def predict_outbreak(
    latitude: float = Form(...),
    longitude: float = Form(...),
    description: str = Form(""),
    native_language: str = Form("English"),
    explicit_crop_type: str = Form("Unknown"),
    crop_image: UploadFile = File(...),
    soil_image: UploadFile = File(None),
    crop_age_days: int = Form(45),
    db: Session = Depends(get_db)
):
    # Validation for mandatory location
    if latitude is None or longitude is None or (latitude == 0.0 and longitude == 0.0):
        raise HTTPException(
            status_code=400, 
            detail="Location data is mandatory. Please enable location access in your browser."
        )

    try:
        # Step 1: Save Images
        crop_image_path = None
        soil_image_path = None
        timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
        
        if crop_image and crop_image.filename:
            crop_ext = crop_image.filename.split('.')[-1]
            crop_filename = f"crop_{timestamp_str}.{crop_ext}"
            crop_image_path = os.path.join(uploads_dir, crop_filename)
            with open(crop_image_path, "wb") as buffer:
                shutil.copyfileobj(crop_image.file, buffer)
                
        if soil_image and soil_image.filename:
            soil_ext = soil_image.filename.split('.')[-1]
            soil_filename = f"soil_{timestamp_str}.{soil_ext}"
            soil_image_path = os.path.join(uploads_dir, soil_filename)
            with open(soil_image_path, "wb") as buffer:
                shutil.copyfileobj(soil_image.file, buffer)

        # STAGE 1: DIAGNOSIS (Vision + LLM Analysis)
        diagnosis_features = await extract_ml_features(
            description=description, 
            crop_image_path=crop_image_path, 
            soil_image_path=soil_image_path,
            latitude=latitude,
            longitude=longitude,
            explicit_crop_type=explicit_crop_type,
            crop_age_days=crop_age_days
        )
        disease_type = diagnosis_features.get('disease_type', 'Unknown')
        severity = diagnosis_features.get('severity', 0.0)
        crop_type = diagnosis_features.get('crop_type', 'unknown').lower()
        
        # STAGE 2: ENVIRONMENT (Weather & Forecast)
        weather_data = await get_current_weather(latitude, longitude)
        
        # Combine all 11 features for ML
        full_features = {**diagnosis_features, **weather_data}
        
        # STAGE 3: FEATURE BUILDER & ML VERIFICATION (Risk Prediction)
        supported_crops = get_supported_crops()
        is_supported = crop_type in supported_crops
        
        if is_supported:
            ml_result = predict_crop_disease(full_features)
            future_severity = ml_result['future_severity']
            risk_level = ml_result['risk_level']
            final_disease = ml_result['verified_disease']
        else:
            # Fallback for unsupported crops
            future_severity = severity 
            risk_level = "High" if severity > 0.7 else "Medium" if severity > 0.4 else "Low"
            final_disease = disease_type

        # STAGE 4: LLM RECOMMENDATION ENGINE (Prognostic Insights)
        recommendation = await generate_recommendation(
            description=description,
            crop_image_path=crop_image_path,
            soil_image_path=soil_image_path,
            weather_data=weather_data,
            ml_disease=final_disease,
            ml_risk_level=risk_level,
            native_language=native_language
        )
        
        # Step 7: Save to Database
        db_record = models.PredictionRecord(
            description=description,
            crop_image_path=crop_image_path,
            soil_image_path=soil_image_path,
            native_language=native_language,
            latitude=latitude,
            longitude=longitude,
            disease_type=disease_type,
            severity=severity,
            infection_area=diagnosis_features.get('infection_area'),
            crop_age_days=crop_age_days,
            crop_type=crop_type,
            temperature=weather_data.get('temperature'),
            humidity=weather_data.get('humidity'),
            rainfall=weather_data.get('precipitation'),
            wind_speed=weather_data.get('wind_speed'),
            solar_radiation=weather_data.get('solar_radiation'),
            pressure=weather_data.get('pressure'),
            outbreak_trend=weather_data.get('outbreak_trend'),
            forecast_summary=weather_data.get('forecast_summary'),
            future_severity=future_severity,
            risk_level=risk_level,
            verified_disease=final_disease,
            llm_recommendations=recommendation
        )
        db.add(db_record)
        db.commit()
        db.refresh(db_record)
        
        return {
            "id": db_record.id,
            "timestamp": db_record.timestamp,
            "disease_type": disease_type,
            "severity": severity,
            "verified_disease": final_disease,
            "future_severity": future_severity,
            "risk_level": risk_level,
            "temperature": weather_data.get('temperature'),
            "humidity": weather_data.get('humidity'),
            "rainfall": weather_data.get('precipitation'),
            "wind_speed": weather_data.get('wind_speed'),
            "outbreak_trend": weather_data.get('outbreak_trend'),
            "forecast": weather_data.get('forecast_summary'),
            "llm_recommendations": recommendation,
            "is_supported": is_supported
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)

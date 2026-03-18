from fastapi import FastAPI, Depends, HTTPException, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import os
import shutil
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
    crop_image: UploadFile = File(None),
    soil_image: UploadFile = File(None),
<<<<<<< HEAD
    crop_age_days: int = Form(45),
=======
>>>>>>> 553f058264775d7b5ec84062c5f75407d324eb83
    db: Session = Depends(get_db)
):
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

<<<<<<< HEAD
        # STAGE 1: DIAGNOSIS (Vision + LLM Analysis)
        diagnosis_features = await extract_ml_features(
=======
        # Step 2: Extract Features using Qwen2-VL
        extracted_features = await extract_ml_features(
>>>>>>> 553f058264775d7b5ec84062c5f75407d324eb83
            description=description, 
            crop_image_path=crop_image_path, 
            soil_image_path=soil_image_path,
            latitude=latitude,
            longitude=longitude,
<<<<<<< HEAD
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
=======
            explicit_crop_type=explicit_crop_type
        )
        
        # Step 3: Fetch Weather Data
        weather_data = await get_current_weather(latitude, longitude)
        
        # Step 4: Agentic Validation (Pre-ML Check)
        crop_type = extracted_features.get('crop_type', 'unknown').lower()
        supported_crops = get_supported_crops()
        
        if crop_type not in supported_crops:
            # Bypass ML logic and generate Fallback recommendation directly
            fallback_rec = await generate_unsupported_crop_recommendation(
                crop_type=crop_type,
                description=description,
                weather_data=weather_data,
                native_language=native_language
            )
            
            # Save to new Unsupported table for future ML training loop
            unsupported_record = models.UnsupportedCropRecord(
                description=description,
                crop_image_path=crop_image_path,
                soil_image_path=soil_image_path,
                native_language=native_language,
                extracted_crop_type=crop_type,
                extracted_soil_type=extracted_features.get('soil_type'),
                latitude=latitude,
                longitude=longitude,
                llm_estimate=fallback_rec
            )
            db.add(unsupported_record)
            db.commit()
            db.refresh(unsupported_record)
            
            return {
                "id": unsupported_record.id,
                "timestamp": unsupported_record.timestamp,
                "predicted_disease": f"Unknown (Model untrained on {crop_type})",
                "probability": 0.0,
                "risk_level": "Fallback Estimate",
                "temperature": weather_data.get('temperature'),
                "humidity": weather_data.get('humidity'),
                "llm_recommendations": fallback_rec,
                "is_supported": False
            }
            
        # Step 5: Run Machine Learning Model (XGBoost)
        # Note: XGBoost is configured to take structured features, but we will eventually train it on weather too.
        # For now, we pass the extracted features.
        ml_result = predict_crop_disease(extracted_features)
        
        # Step 6: Run LLM for insights (Llama 3.1 / Qwen2.5)
>>>>>>> 553f058264775d7b5ec84062c5f75407d324eb83
        recommendation = await generate_recommendation(
            description=description,
            crop_image_path=crop_image_path,
            soil_image_path=soil_image_path,
            weather_data=weather_data,
<<<<<<< HEAD
            ml_disease=final_disease,
            ml_risk_level=risk_level,
=======
            ml_disease=ml_result['predicted_disease'],
            ml_risk_level=ml_result['risk_level'],
>>>>>>> 553f058264775d7b5ec84062c5f75407d324eb83
            native_language=native_language
        )
        
        # Step 7: Save to Database
        db_record = models.PredictionRecord(
            description=description,
            crop_image_path=crop_image_path,
            soil_image_path=soil_image_path,
            native_language=native_language,
<<<<<<< HEAD
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
=======
            temperature=weather_data.get('temperature'),
            humidity=weather_data.get('humidity'),
            predicted_disease=ml_result['predicted_disease'],
            probability=ml_result['probability'],
            risk_level=ml_result['risk_level'],
            llm_recommendations=recommendation,
            **{k: v for k, v in extracted_features.items() if k.startswith('attr_')}
>>>>>>> 553f058264775d7b5ec84062c5f75407d324eb83
        )
        db.add(db_record)
        db.commit()
        db.refresh(db_record)
        
        return {
            "id": db_record.id,
            "timestamp": db_record.timestamp,
<<<<<<< HEAD
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
=======
            "predicted_disease": db_record.predicted_disease,
            "probability": db_record.probability,
            "risk_level": db_record.risk_level,
            "temperature": db_record.temperature,
            "humidity": db_record.humidity,
            "llm_recommendations": db_record.llm_recommendations,
            "is_supported": True
>>>>>>> 553f058264775d7b5ec84062c5f75407d324eb83
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

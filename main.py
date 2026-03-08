from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import os

from db.database import Base, engine, get_db
from db import models, schemas
from services.weather_service import get_current_weather
from services.ml_service import predict_crop_disease
from services.llm_service import generate_recommendation

# Create db tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Crop Disease Outbreak Prediction System",
    description="Agentic Backend System for predicting crop diseases and offering LLM recommendations."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for UI
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def root():
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.post("/predict", response_model=schemas.PredictionResponse)
async def predict_outbreak(request: schemas.PredictionCreate, db: Session = Depends(get_db)):
    try:
        # Step 1: Extract features for ML model (everything except lat/lon)
        features_dict = request.model_dump(exclude={'latitude', 'longitude'})
        
        # Step 2: Fetch Weather Data
        weather_data = await get_current_weather(request.latitude, request.longitude)
        
        # Step 3: Run Machine Learning Model (XGBoost)
        ml_result = predict_crop_disease(features_dict)
        
        # Step 4: Run Agentic LLM for insights
        recommendation = await generate_recommendation(
            disease=ml_result['predicted_disease'],
            risk_level=ml_result['risk_level'],
            temp=weather_data.get('temperature'),
            humidity=weather_data.get('humidity')
        )
        
        # Step 5: Save to Database
        db_record = models.PredictionRecord(
            **features_dict,
            temperature=weather_data.get('temperature'),
            humidity=weather_data.get('humidity'),
            predicted_disease=ml_result['predicted_disease'],
            probability=ml_result['probability'],
            risk_level=ml_result['risk_level'],
            llm_recommendations=recommendation
        )
        db.add(db_record)
        db.commit()
        db.refresh(db_record)
        
        return db_record
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

from sqlalchemy import Column, Integer, String, Float, DateTime
from db.database import Base
import datetime

class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    # User Inputs
    description = Column(String, nullable=True)
    crop_image_path = Column(String, nullable=True)
    soil_image_path = Column(String, nullable=True)
    native_language = Column(String, default="English")
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

    # Phase 1: Diagnosis (LLM Observations)
    disease_type = Column(String, nullable=True)
    severity = Column(Float, nullable=True)
    infection_area = Column(Float, nullable=True)
    crop_age_days = Column(Integer, nullable=True)
    crop_type = Column(String, nullable=True)

    # Phase 2: Weather & Environment
    temperature = Column(Float, nullable=True)
    humidity = Column(Float, nullable=True)
    rainfall = Column(Float, nullable=True)
    wind_speed = Column(Float, nullable=True)
    solar_radiation = Column(Float, nullable=True)
    pressure = Column(Float, nullable=True)
    outbreak_trend = Column(String, nullable=True)
    forecast_summary = Column(String, nullable=True)
    
    # Phase 3: ML Verification & Outbreak Prognosis
    future_severity = Column(Float) # Target prediction
    risk_level = Column(String)
    verified_disease = Column(String, index=True) 
    
    # Phase 4: LLM Insights
    llm_recommendations = Column(String, nullable=True)

class UnsupportedCropRecord(Base):
    __tablename__ = "unsupported_crops"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    description = Column(String, nullable=True)
    crop_image_path = Column(String, nullable=True)
    soil_image_path = Column(String, nullable=True)
    native_language = Column(String, default="English")
    
    extracted_crop_type = Column(String, nullable=True)
    extracted_soil_type = Column(String, nullable=True)
    
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    
    llm_estimate = Column(String, nullable=True)

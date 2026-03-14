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

    # Extracted/Historical Features (from Qwen2-VL or UI usually)
    attr_1 = Column(Float, nullable=True)
    attr_2 = Column(Float, nullable=True)
    attr_3 = Column(Float, nullable=True)
    attr_4 = Column(Float, nullable=True)
    attr_5 = Column(Float, nullable=True)
    attr_6 = Column(Float, nullable=True)
    attr_7 = Column(Float, nullable=True)
    attr_8 = Column(Float, nullable=True)
    attr_9 = Column(Float, nullable=True)
    attr_10 = Column(Float, nullable=True)
    attr_35 = Column(Float, nullable=True)
    
    # Weather
    temperature = Column(Float, nullable=True)
    humidity = Column(Float, nullable=True)
    
    # ML Outputs
    predicted_disease = Column(String, index=True)
    probability = Column(Float)
    risk_level = Column(String)
    
    # LLM Insights
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

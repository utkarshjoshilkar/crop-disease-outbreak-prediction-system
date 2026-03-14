from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class PredictionCreate(BaseModel):
    description: Optional[str] = None
    native_language: str = "English"
    
    # Coordinates to fetch weather
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class PredictionResponse(BaseModel):
    id: int
    timestamp: datetime
    predicted_disease: str
    probability: float
    risk_level: str
    temperature: Optional[float]
    humidity: Optional[float]
    llm_recommendations: Optional[str]
    is_supported: bool = True
    
    class Config:
        from_attributes = True

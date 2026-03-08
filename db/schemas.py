from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class PredictionCreate(BaseModel):
    attr_1: float
    attr_2: float
    attr_3: float
    attr_4: float
    attr_5: float
    attr_6: float
    attr_7: float
    attr_8: float
    attr_9: float
    attr_10: float
    attr_35: float
    
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
    
    class Config:
        from_attributes = True

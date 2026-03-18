from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class PredictionCreate(BaseModel):
    description: Optional[str] = None
    native_language: str = "English"
    
    # Coordinates to fetch weather
    latitude: Optional[float] = None
    longitude: Optional[float] = None
<<<<<<< HEAD
    crop_age_days: Optional[int] = 45
=======
>>>>>>> 553f058264775d7b5ec84062c5f75407d324eb83

class PredictionResponse(BaseModel):
    id: int
    timestamp: datetime
<<<<<<< HEAD
    disease_type: str
    future_severity: float
    risk_level: str
    temperature: Optional[float]
    humidity: Optional[float]
    rainfall: Optional[float]
    wind_speed: Optional[float]
    outbreak_trend: Optional[str]
=======
    predicted_disease: str
    probability: float
    risk_level: str
    temperature: Optional[float]
    humidity: Optional[float]
>>>>>>> 553f058264775d7b5ec84062c5f75407d324eb83
    llm_recommendations: Optional[str]
    is_supported: bool = True
    
    class Config:
        from_attributes = True

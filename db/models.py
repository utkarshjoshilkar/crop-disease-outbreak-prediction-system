from sqlalchemy import Column, Integer, String, Float, DateTime
from db.database import Base
import datetime

class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    # User Input Features (example map names to generic for now, typically these match UI)
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

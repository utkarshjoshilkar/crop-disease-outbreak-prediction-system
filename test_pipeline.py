import asyncio
from services.weather_service import get_current_weather
from services.ml_service import predict_crop_disease
from services.llm_service import generate_recommendation

async def test_pipeline():
    print("--- Testing Agentic Crop Disease Pipeline ---")
    
    # Mock user input
    lat = 28.6139 # New Delhi
    lon = 77.2090
    features = {
        "attr_1": 1.0,
        "attr_2": 0.5,
        "attr_3": 0.0,
        "attr_4": 0.0,
        "attr_5": 0.0,
        "attr_6": 0.0,
        "attr_7": 0.0,
        "attr_8": 0.0,
        "attr_9": 0.0,
        "attr_10": 0.0,
        "attr_35": 0.0
    }
    
    # 1. Weather
    print("\n1. Fetching Weather Data...")
    weather = await get_current_weather(lat, lon)
    print(f"Weather Result: {weather}")
    
    # 2. ML Model
    print("\n2. Running XGBoost Model...")
    ml_result = predict_crop_disease(features)
    print(f"ML Result: {ml_result}")
    
    # 3. LLM Agent
    print("\n3. Generating Agentic Recommendations...")
    llm_result = await generate_recommendation(
        disease=ml_result['predicted_disease'],
        risk_level=ml_result['risk_level'],
        temp=weather.get('temperature'),
        humidity=weather.get('humidity')
    )
    print(f"LLM Insights: {llm_result}")

if __name__ == "__main__":
    asyncio.run(test_pipeline())

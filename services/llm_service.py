import logging

logger = logging.getLogger(__name__)

async def generate_recommendation(disease: str, risk_level: str, temp: float, humidity: float) -> str:
    """
    Simulates a call to an LLM to generate recommendations based on the prediction and environmental factors.
    Replace this with an actual OpenAI / Gemini / Groq API call.
    """
    logger.info(f"Generating LLM insights for {disease} with {risk_level} risk.")
    
    # Mock LLM generation
    if risk_level == "High":
        insight = f"URGENT: High risk of {disease} detected. Immediate action required. "
        insight += "Apply appropriate fungicides immediately and isolate affected sections if possible. "
    elif risk_level == "Medium":
        insight = f"WARNING: Medium risk of {disease}. Monitor crops closely over the next 48 hours. "
        insight += "Consider preventative spraying if conditions worsen. "
    else:
        insight = f"Good news. Low risk of {disease} detected. Maintain standard farming practices. "
        
    if temp and temp > 30 and humidity and humidity > 70:
        insight += f"Note: Current high temperature ({temp}°C) and high humidity ({humidity}%) create an ideal environment for fungal and bacterial spread. Enhance ventilation and monitor moisture."
    elif temp and humidity:
        insight += f"Current conditions: Temp {temp}°C, Humidity {humidity}%."
        
    return insight

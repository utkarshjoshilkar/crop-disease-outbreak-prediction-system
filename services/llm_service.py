import logging
from typing import Dict, Optional, Any
from deep_translator import GoogleTranslator

logger = logging.getLogger(__name__)

async def extract_ml_features(
    description: Optional[str], 
    crop_image_path: Optional[str], 
    soil_image_path: Optional[str],
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    explicit_crop_type: str = "Unknown"
) -> Dict[str, Any]:
    """
    Uses Qwen2-VL (Vision Model) to extract structural numerical features based on images and descriptions.
    If soil image/text is missing, it estimates soil type using latitude/longitude.
    """
    logger.info(f"Extracting ML features using Qwen2-VL. Inputs provided: Desc={bool(description)}, CropImg={bool(crop_image_path)}, SoilImg={bool(soil_image_path)}")
    
    # Mock Qwen2-VL extraction returning required attributes for the ML model
    
    # Simple keyword mock logic for testing:
    desc_lower = (description or "").lower()
    img_path_lower = (crop_image_path or "").lower()
    
    # Simulated Vision/Text extraction logic:
    if explicit_crop_type and explicit_crop_type.lower() not in ["unknown", "other", ""]:
        mock_crop_type = explicit_crop_type.lower()
    elif "cotton" in desc_lower or "cotton" in img_path_lower:
        mock_crop_type = "cotton"
    elif "wheat" in desc_lower or "wheat" in img_path_lower:
        mock_crop_type = "wheat"
    elif "rice" in desc_lower or "rice" in img_path_lower:
        mock_crop_type = "rice"
    elif "tomato" in desc_lower or "tomato" in img_path_lower:
        mock_crop_type = "tomato"
    elif "soybean" in desc_lower or "soya bean" in desc_lower or "soybean" in img_path_lower:
        mock_crop_type = "soybean"
    else:
        # If the vision model and text model can't identify it distinctly, default to soybean for this mock
        mock_crop_type = "soybean"
        
    mock_soil_type = "alluvial" if not soil_image_path else "black soil"
    
    mock_features = {
        'attr_1': 0.0, 'attr_2': 0.0, 'attr_3': 0.0, 'attr_4': 0.0, 'attr_5': 0.0,
        'attr_6': 0.0, 'attr_7': 0.0, 'attr_8': 0.0, 'attr_9': 0.0, 'attr_10': 0.0,
        'attr_35': 0.0,
        'crop_type': mock_crop_type,
        'soil_type': mock_soil_type
    }
    return mock_features

async def generate_recommendation(
    description: Optional[str],
    crop_image_path: Optional[str], 
    soil_image_path: Optional[str],
    weather_data: Dict[str, Any],
    ml_disease: str,
    ml_risk_level: str,
    native_language: str
) -> str:
    """
    Simulates a call to Llama 3.1 / Qwen2.5 to generate farmer-friendly recommendations.
    Replace this with an actual Groq / HuggingFace API call.
    """
    logger.info(f"Generating Llama/Qwen2.5 insights in {native_language} for {ml_disease} ({ml_risk_level} risk).")
    
    temp = weather_data.get('temperature')
    humidity = weather_data.get('humidity')

    # Constructing a prompt that would be sent to the LLM
    prompt_used = (
        f"You are an expert agronomist. The farmer describes the crop as: '{description}'. "
        f"The ML model predicts '{ml_disease}' with a '{ml_risk_level}' risk level. "
        f"Weather is {temp}°C and {humidity}% humidity. "
        f"Generate actionable advice in simple layman terms in the language: {native_language}."
    )
    
    # Mock Response Logic simulating the LLM Output
    insight_prefix = f"*** Translated to {native_language} (Simple Farmer Terms) ***\n\n"
    
    if ml_risk_level == "High":
        insight = f"{insight_prefix}URGENT: High risk of {ml_disease} detected. You need to take action right away to save your crop. "
        insight += "Please spray the correct fungus medicine immediately and try to separate the sick plants from the healthy ones. "
    elif ml_risk_level == "Medium":
        insight = f"{insight_prefix}WARNING: There is a medium chance of {ml_disease}. Keep a very close eye on your plants for the next 48 hours. "
        insight += "If the spots or wilting gets worse, you should spray a preventative medicine. "
    else:
        insight = f"{insight_prefix}Good news! There is a very low chance of {ml_disease}. Keep up your normal farming routine. "
        
        insight += f"\nNote: The high heat ({temp}°C) and moisture ({humidity}%) right now makes it very easy for sickness to spread quickly in the field."
        
    insight += f"\n\n[Agentic AI Insights provided by Llama 3.1 / Qwen2.5]"

    if native_language.lower() != "english":
        try:
            lang_map = {
                "hindi": "hi", "marathi": "mr", "telugu": "te", 
                "tamil": "ta", "gujarati": "gu", "spanish": "es"
            }
            base_lang = native_language.split()[0].lower()
            lang_code = lang_map.get(base_lang, "en")
            if lang_code != "en":
                insight = GoogleTranslator(source='auto', target=lang_code).translate(insight)
        except Exception as e:
            logger.error(f"Translation failed: {e}")

    return insight

async def generate_unsupported_crop_recommendation(
    crop_type: str,
    description: Optional[str],
    weather_data: Dict[str, Any],
    native_language: str
) -> str:
    """
    Generates a fallback recommendation when the ML model has not been trained on the identified crop type.
    """
    logger.info(f"Generating LLM Fallback for unsupported crop: {crop_type} in {native_language}")
    
    insight = (
        f"*** Translated to {native_language} (Simple Farmer Terms) ***\n\n"
        f"Right now, our main computer system doesn't have enough past data on '{crop_type}' to give you a 100% certain disease scan. "
        f"But looking at what you told us ('{description}'), this seems like a common fungal sickness or your soil might be missing some food nutrients.\n\n"
        f"As a safe guess, try to make sure water isn't gathering around the plant roots. It is a very good idea to show these signs to your local farming expert or seed shop just to be safe.\n\n"
        f"We have safely saved your picture and description. We will use this to teach our AI system to handle '{crop_type}' for you in the next update!"
    )
    
    insight += f"\n\n[LLM ZERO-SHOT ESTIMATE - Generated by Llama 3.1]"
    
    if native_language.lower() != "english":
        try:
            lang_map = {
                "hindi": "hi", "marathi": "mr", "telugu": "te", 
                "tamil": "ta", "gujarati": "gu", "spanish": "es"
            }
            base_lang = native_language.split()[0].lower()
            lang_code = lang_map.get(base_lang, "en")
            if lang_code != "en":
                insight = GoogleTranslator(source='auto', target=lang_code).translate(insight)
        except Exception as e:
            logger.error(f"Translation failed: {e}")

    return insight

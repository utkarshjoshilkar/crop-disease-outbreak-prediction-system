import logging
import json
import httpx
import base64
from typing import Dict, Optional, Any, List
from deep_translator import GoogleTranslator
from config import LLM_BASE_URL, LLM_MODEL, VISION_MODEL

logger = logging.getLogger(__name__)

def _encode_image(image_path: str) -> str:
    """Encodes a local image to base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        return ""

async def _call_local_llm(prompt: str, system_prompt: str = "You are an expert agronomist.", model: str = LLM_MODEL, images: Optional[List[str]] = None) -> str:
    """
    Calls the local Ollama API to generate a response.
    """
    try:
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
            }
        }
        
        if images:
            payload["images"] = images
            
        # DEBUG PRINT
        print(f"--- Sending request to Ollama ({model}): {LLM_BASE_URL} ---")
        
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(LLM_BASE_URL, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "No response from LLM").strip()
            
    except Exception as e:
        logger.error(f"Error calling local LLM: {repr(e)}")
        print(f"--- FAILED TO CALL LLM ({model}): {repr(e)} ---")
        return f"Error: Could not reach Vision/LLM. Please ensure Ollama is running and '{model}' is pulled. (Technical detail: {repr(e)})"


async def extract_ml_features(
    description: Optional[str], 
    crop_image_path: Optional[str], 
    soil_image_path: Optional[str],
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    explicit_crop_type: str = "Unknown",
    crop_age_days: int = 45
) -> Dict[str, Any]:
    """
    PHASE 1: DIAGNOSIS
    Uses Vision-LLM (Llava) or Text-LLM (Llama 3.1) to identify disease and severity.
    """
    logger.info(f"Extracting ML features and Diagnosis. Inputs: Desc={bool(description)}, CropImg={bool(crop_image_path)}")
    
    images_to_send = []
    if crop_image_path:
        encoded = _encode_image(crop_image_path)
        if encoded: images_to_send.append(encoded)
    if soil_image_path:
        encoded = _encode_image(soil_image_path)
        if encoded: images_to_send.append(encoded)

    llm_raw = ""

    if images_to_send or description:
        model_to_use = VISION_MODEL if images_to_send else LLM_MODEL
        prompt = (
            f"Crop: '{explicit_crop_type}'. Description: '{description}'. "
            "Task: Analyze the images (if any) and description to extract features in JSON. "
            "Format exactly as: {'crop_type': string, 'disease_type': string, 'severity': float (0.0 to 1.0), 'infection_area': float (0.0 to 1.0)}. "
            "Output ONLY valid JSON. No markdown blocks, no conversational text."
        )
        
        system_prompt = "You are a specialized agricultural vision-language model. Output strictly in JSON format."
        
        try:
            llm_raw = await _call_local_llm(prompt, system_prompt, model=model_to_use, images=images_to_send)
            
            # Clean possible markdown blocks and whitespace
            clean_json = llm_raw.replace('```json', '').replace('```', '').strip()
            import re
            json_match = re.search(r'\{.*\}', clean_json, re.DOTALL)
            if json_match:
                extracted = json.loads(json_match.group(0))
                # Ensure severity and infection_area are floats, not strings
                try: severity = float(extracted.get('severity', 0.5))
                except: severity = 0.5
                try: area = float(extracted.get('infection_area', 0.1))
                except: area = 0.1
                
                return {
                    'crop_type': extracted.get('crop_type', explicit_crop_type).lower(),
                    'crop_age_days': crop_age_days,
                    'disease_type': extracted.get('disease_type', 'Unknown'),
                    'severity': severity,
                    'infection_area': area,
                    'diagnosis_probability': 0.85 if images_to_send else 0.75
                }
        except Exception as e:
            logger.error(f"Failed to parse LLM diagnosis JSON from '{llm_raw}': {e}")
            # Do NOT return default immediately. Let it fall through to text analysis.

    # Fallback: Parse the raw LLM output text or user description for keywords
    text_to_analyze = (llm_raw + " " + (description or "")).lower()
    img_path_lower = (crop_image_path or "").lower()
    
    # 1. Deduce crop type
    if explicit_crop_type and explicit_crop_type.lower() not in ["unknown", "other", ""]:
        crop_type = explicit_crop_type.lower()
    elif ("maize" in text_to_analyze or "corn" in text_to_analyze):
        crop_type = "maize"
    elif "wheat" in text_to_analyze:
        crop_type = "wheat"
    elif "rice" in text_to_analyze:
        crop_type = "rice"
    else:
        crop_type = "wheat" # safe default
        
    # 2. Deduce disease type from LLM raw text / description
    disease_indicators = ["yellow", "spot", "brown_spot", "blight", "rust", "blast", "wilt", "disease", "dry", "lesion", "rot"]
    if any(k in text_to_analyze for k in disease_indicators):
        if "blast" in text_to_analyze:
            disease_type = "Blast"; severity = 0.70; area = 0.35
        elif "blight" in text_to_analyze:
            disease_type = "Leaf Blight"; severity = 0.70; area = 0.30
        elif "rust" in text_to_analyze:
            disease_type = "Rust"; severity = 0.60; area = 0.25
        elif "spot" in text_to_analyze or "yellow" in text_to_analyze or "lesion" in text_to_analyze:
            disease_type = "Brown Spot"; severity = 0.65; area = 0.25
        elif "wilt" in text_to_analyze or "dry" in text_to_analyze:
            disease_type = "Fusarium Wilt"; severity = 0.80; area = 0.40
        elif "rot" in text_to_analyze:
            disease_type = "Rot"; severity = 0.90; area = 0.50
        else:
            disease_type = "General Disease"; severity = 0.50; area = 0.20
    else:
        # Only true if the LLM positively identified it as healthy and found no spots
        disease_type = "Healthy"; severity = 0.10; area = 0.05

    return {
        'crop_type': crop_type,
        'crop_age_days': crop_age_days,
        'disease_type': disease_type,
        'severity': severity,
        'infection_area': area,
        'diagnosis_probability': 0.60
    }

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
    Calls Llama 3.1 to generate farmer-friendly recommendations.
    """
    logger.info(f"Generating Llama 3.1 insights in {native_language} for {ml_disease} ({ml_risk_level} risk).")
    
    temp = weather_data.get('temperature')
    humidity = weather_data.get('humidity')

    # Constructing a prompt that would be sent to the LLM
    prompt = (
        f"The farmer describes the crop as: '{description}'. "
        f"The ML model predicts '{ml_disease}' with a '{ml_risk_level}' risk level. "
        f"Weather is {temp}°C and {humidity}% humidity. "
        f"Generate actionable advice in simple layman terms for the farmer."
    )
    
    system_prompt = (
        "You are an expert agronomist. Provide professional, encouraging, and clear advice. "
        "Focus on immediate steps the farmer can take. Respond directly with the advice, "
        "no conversational filler unless helpful."
    )

    # Call the actual LLM
    insight = await _call_local_llm(prompt, system_prompt)
    
    # Prepend translation note if it's in English (which it is by default)
    # Actually, we will translate it AFTER if needed.
    
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

    return f"*** Advice in {native_language} ***\n\n{insight}\n\n[Real-time Insights from Llama 3.1 8B]"

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
    
    prompt = (
        f"The farmer has an unsupported crop: '{crop_type}'. "
        f"Description: '{description}'. "
        f"Current weather: {weather_data}. "
        f"Provide a general estimate and safety advice since the ML model is not specifically trained for this crop."
    )
    
    system_prompt = (
        "You are an expert agronomist. Provide helpful general advice for crops not in our primary database. "
        "Be cautious but helpful."
    )

    insight = await _call_local_llm(prompt, system_prompt)
    
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

    return f"*** General Advice in {native_language} ***\n\n{insight}\n\n[Llama 3.1 Zero-Shot Estimate]"

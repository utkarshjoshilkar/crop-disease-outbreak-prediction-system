import joblib
import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier

# Get absolute path to the models directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

class XGBWrapper:
    def __init__(self, **kwargs):
        self.model = XGBClassifier(**kwargs)
        self.classes_ = None
        self.label_map = {}
        self.inverse_label_map = {}
        
    def fit(self, X, y):
        pass # Not needed for inference
        
    def predict(self, X):
        y_pred = self.model.predict(X)
        return np.array([self.inverse_label_map[val] for val in y_pred])
        
    def predict_proba(self, X):
        probas = self.model.predict_proba(X)
        full_probas = np.zeros((len(X), 19))
        for i, c in enumerate(self.classes_):
            full_probas[:, c] = probas[:, i]
        return full_probas

# Need to inject the wrapper into __main__ before loading because it was pickled that way
import __main__
if not hasattr(__main__, 'XGBWrapper'):
    __main__.XGBWrapper = XGBWrapper

# Load Models
try:
    model = joblib.load(os.path.join(MODELS_DIR, 'model.joblib'))
    encoders = joblib.load(os.path.join(MODELS_DIR, 'encoders.joblib'))
    target_encoder = joblib.load(os.path.join(MODELS_DIR, 'target_encoder.joblib'))
except Exception as e:
    print(f"Error loading models: {e}. Please ensure the models exist in the models/ directory.")
    model, encoders, target_encoder = None, None, None

def calculate_risk_level(p: float) -> str:
    if p < 0.4:
        return "Low"
    elif p < 0.7:
        return "Medium"
    else:
        return "High"

def predict_crop_disease(features: dict) -> dict:
    if model is None:
        return {"predicted_disease": "Error: Model not loaded", "probability": 0.0, "risk_level": "Unknown"}
        
    # Isolate only the numerical ML attributes the model was trained on
    ml_features = {k: v for k, v in features.items() if k.startswith('attr_')}
    df = pd.DataFrame([ml_features])
    
    # Process categorical encodings if they apply
    # (Assuming feature names match exactly what was trained)
    for col in df.columns:
        if col in encoders:
            le = encoders[col]
            try:
                df[col] = le.transform(df[col])
            except ValueError:
                df[col] = 0 # Fallback for unknown classes
                
    # Predict
    probs = model.predict_proba(df)[0]
    max_prob = float(np.max(probs))
    pred_idx = int(np.argmax(probs))
    
    # Decode result
    try:
        pred_disease = target_encoder.inverse_transform([pred_idx])[0]
    except Exception:
        pred_disease = f"Class_{pred_idx}"
        
    risk = calculate_risk_level(max_prob)
    
    return {
        "predicted_disease": str(pred_disease),
        "probability": max_prob,
        "risk_level": risk
    }

import json

def get_supported_crops() -> list:
    """Returns the list of crops the XGBoost model is currently trained to predict."""
    config_path = os.path.join(BASE_DIR, "disease_prediction_system", "supported_crops.json")
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                data = json.load(f)
                return [c.lower() for c in data.get("supported_crops", [])]
    except Exception as e:
        print(f"Warning: Could not read supported_crops.json -> {e}")
        
    return ["soybean", "soya bean", "soybeans"]

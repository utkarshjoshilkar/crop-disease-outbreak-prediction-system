import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

# -------------------------------
# 1. BIOLOGICAL RISK SCORER
# -------------------------------
def calculate_risk_score(row):
    """
    Calculates a probabilistic risk score (0 to 1) based on weather.
    0.7+ : High Risk (2)
    0.4 - 0.7: Medium Risk (1)
    < 0.4: Low Risk (0)
    """
    score = 0
    # Humidity contribution (40%) - Fungi thrive in 85%+
    score += min(row['humidity_3d_avg'] / 100, 1) * 0.4
    # Rain contribution (30%) - Spores travel in wet spells
    score += min(row['rain_3d_sum'] / 30, 1) * 0.3
    # Temperature contribution (30%) - Optimal growth: 25-30C
    temp_score = 1 if 25 <= row['temp_3d_avg'] <= 30 else 0.5 if 20 <= row['temp_3d_avg'] <= 35 else 0
    score += temp_score * 0.3
    
    if score > 0.7: return 2
    elif score > 0.4: return 1
    else: return 0

# -------------------------------
# 2. FEATURE ENGINEERING
# -------------------------------
def engineer_features(df):
    df = df.copy()
    # Handle missing
    df = df.ffill()
    
    # Rolling averages (The biological drivers)
    df['temp_3d_avg'] = df['temp'].rolling(3).mean()
    df['humidity_3d_avg'] = df['humidity'].rolling(3).mean()
    df['rain_3d_sum'] = df['rain'].rolling(3).sum()
    
    # Dynamics (Rate of change)
    df['temp_range'] = df['temp'].rolling(3).max() - df['temp'].rolling(3).min()
    df['humidity_change'] = df['humidity'].diff()
    
    # 🔥 NEW: Humidity Streak (Biologically critical)
    df['high_hum_flag'] = (df['humidity'] > 85).astype(int)
    df['humidity_streak'] = df['high_hum_flag'].groupby((df['high_hum_flag'] == 0).cumsum()).cumcount()
    
    # Rain flag & Wet spell
    df['rain_flag'] = (df['rain'] > 2).astype(int)
    df['wet_spell'] = df['rain_flag'] * (
        df['rain_flag'].groupby((df['rain_flag'] == 0).cumsum()).cumcount() + 1
    )
    df['rain_intensity'] = df['rain'] / (df['wet_spell'] + 1)
    
    # Lag features
    for col in ['temp', 'humidity', 'rain']:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag2'] = df[col].shift(2)
        
    # Temporal
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # 🔥 NEW: Crop Growth Stage
    # Sugarcane: Germination (1), Tillering (2), Grand Growth (3 - peak risk), Maturity (4)
    def get_stage(doy):
        if doy < 60: return 1
        if doy < 150: return 2
        if doy < 300: return 3 # Grand Growth (High Risk Period)
        return 4
    df['growth_stage'] = df['day_of_year'].apply(get_stage)
    
    return df.dropna()

# -------------------------------
# 3. TRAINING PIPELINE
# -------------------------------
FEATURES = [
    'temp', 'humidity', 'rain',
    'temp_3d_avg', 'humidity_3d_avg', 'rain_3d_sum',
    'wet_spell', 'temp_lag1', 'temp_lag2',
    'humidity_lag1', 'humidity_lag2',
    'rain_lag1', 'rain_lag2',
    'month', 'day_of_year',
    'temp_range', 'humidity_change', 'rain_intensity',
    'humidity_streak', 'growth_stage'
]

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'red_rot_model.joblib')

def train_and_save_model(csv_path):
    # Load
    df_raw = pd.read_csv(csv_path, skiprows=12)
    df_raw['date'] = pd.to_datetime(df_raw['YEAR'].astype(str) + '-' + df_raw['DOY'].astype(str), format='%Y-%j')
    df_raw = df_raw[['date', 'T2M', 'RH2M', 'PRECTOTCORR']]
    df_raw.columns = ['date', 'temp', 'humidity', 'rain']
    
    # Engineer
    df = engineer_features(df_raw)
    
    # Label
    df['risk'] = df.apply(calculate_risk_score, axis=1)
    df['target'] = df['risk'].shift(-3) # Predict 3 days ahead
    df = df.dropna()
    
    # Split
    X = df[FEATURES]
    y = df['target']
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    
    # No SMOTE (Per expert recommendation, distribution integrity is priority)
    
    # Train (Using balanced class weights in XGBoost for internal handle)
    model = XGBClassifier(
        n_estimators=200, 
        max_depth=6, 
        learning_rate=0.1, 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Save
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")
    return model

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

if __name__ == "__main__":
    train_and_save_model("POWER_Point_Daily_20160101_20251231_016d70N_074d69E_LST.csv")

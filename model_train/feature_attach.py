import pandas as pd
import numpy as np

# Load your engineered dataset
df = pd.read_csv("engineered_weather_features.csv")

# ---------------------------
# 1. DEFINE CONDITIONS (ICAR-INSPIRED)
# ---------------------------

# High humidity (fungal growth)
high_humidity = df['RH2M'] > 80

# Warm temperature (optimal for pathogen)
warm_temp = (df['T2M'] > 25) & (df['T2M'] < 35)

# Recent rainfall (spread + moisture)
recent_rain = df['rain_7d'] > 20

# Monsoon season (India-specific boost)
monsoon = df['month'].isin([6,7,8,9])

# ---------------------------
# 2. CREATE RISK SCORE
# ---------------------------

df['red_rot_score'] = (
    0.4 * high_humidity.astype(int) +
    0.3 * warm_temp.astype(int) +
    0.3 * (df['rain_7d'] / (df['rain_7d'].max() + 1))
)

# Add seasonal boost
df['red_rot_score'] += 0.2 * monsoon.astype(int)

# Normalize
df['red_rot_score'] = df['red_rot_score'].clip(0, 1)

# ---------------------------
# 3. CREATE LABEL
# ---------------------------

# Binary classification
df['red_rot_risk'] = (df['red_rot_score'] > 0.6).astype(int)

# Optional: multi-class (better for real-world)
def risk_category(score):
    if score < 0.3:
        return "Low"
    elif score < 0.6:
        return "Medium"
    else:
        return "High"

df['red_rot_category'] = df['red_rot_score'].apply(risk_category)

# ---------------------------
# 4. SAVE UPDATED DATASET
# ---------------------------

df.to_csv("red_rot_labeled_dataset.csv", index=False)

print("✅ Done! Saved as red_rot_labeled_dataset.csv")
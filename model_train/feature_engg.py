import pandas as pd
import numpy as np

# Load NASA POWER CSV
file_path = "POWER_Point_Daily_20050101_20241231_016d54N_069d78E_LST.csv"

# Skip metadata rows (adjust if needed: 14–16 works for most files)
df = pd.read_csv(file_path, skiprows=14)

# Drop empty rows
df = df.dropna(how='all')

# Create datetime column
df['date'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['DOY'].astype(str), format='%Y-%j')

# Drop original date component columns if they exist
for col in ['YEAR', 'DOY', 'MO', 'DY']:
    if col in df.columns:
        df = df.drop(columns=[col])

# -----------------------
# 🌡️ TEMPERATURE FEATURES
# -----------------------
df['temp_range'] = df['T2M_MAX'] - df['T2M_MIN']
df['temp_mean_3d'] = df['T2M'].rolling(3).mean()
df['temp_mean_7d'] = df['T2M'].rolling(7).mean()
df['temp_mean_14d'] = df['T2M'].rolling(14).mean()

# -----------------------
# 🌧️ RAIN FEATURES
# -----------------------
df['rain_3d'] = df['PRECTOTCORR'].rolling(3).sum()
df['rain_7d'] = df['PRECTOTCORR'].rolling(7).sum()
df['rain_14d'] = df['PRECTOTCORR'].rolling(14).sum()

# Dry / wet flags
df['is_rainy'] = (df['PRECTOTCORR'] > 1).astype(int)
df['dry_spell_7d'] = (df['rain_7d'] < 5).astype(int)

# -----------------------
# 💧 HUMIDITY FEATURES
# -----------------------
df['rh_3d'] = df['RH2M'].rolling(3).mean()
df['rh_7d'] = df['RH2M'].rolling(7).mean()
df['high_humidity'] = (df['RH2M'] > 80).astype(int)

# -----------------------
# 💨 WIND FEATURES
# -----------------------
df['wind_3d'] = df['WS10M'].rolling(3).mean()

# -----------------------
# ⏳ LAG FEATURES (VERY IMPORTANT)
# -----------------------
for lag in [1, 2, 3, 7]:
    df[f'temp_lag_{lag}'] = df['T2M'].shift(lag)
    df[f'rain_lag_{lag}'] = df['PRECTOTCORR'].shift(lag)
    df[f'rh_lag_{lag}'] = df['RH2M'].shift(lag)

# -----------------------
# 📅 TIME FEATURES
# -----------------------
df['month'] = df['date'].dt.month
df['dayofyear'] = df['date'].dt.dayofyear
df['week'] = df['date'].dt.isocalendar().week

# Seasonal encoding (important for ML)
df['sin_day'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
df['cos_day'] = np.cos(2 * np.pi * df['dayofyear'] / 365)

# -----------------------
# 🦠 DISEASE-RISK FEATURES (VERY IMPORTANT FOR YOUR PROJECT)
# -----------------------
df['fungal_risk'] = ((df['RH2M'] > 80) & (df['T2M'] > 20) & (df['PRECTOTCORR'] > 1)).astype(int)

df['heat_stress'] = (df['T2M_MAX'] > 35).astype(int)

df['cold_stress'] = (df['T2M_MIN'] < 10).astype(int)

# -----------------------
# CLEAN DATA
# -----------------------
df = df.dropna()

# Save
df.to_csv("engineered_weather_features.csv", index=False)

print("✅ Done! Saved as engineered_weather_features.csv")
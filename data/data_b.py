# ================================
# 🌾 Sugarcane Red Rot Prediction (Improved)
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -------------------------------
# 1. LOAD DATA
# -------------------------------
df = pd.read_csv(
    "POWER_Point_Daily_20160101_20251231_016d70N_074d69E_LST.csv",
    skiprows=12
)

# Create date column
df['date'] = pd.to_datetime(
    df['YEAR'].astype(str) + '-' + df['DOY'].astype(str),
    format='%Y-%j'
)

# Select columns
df = df[['date', 'T2M', 'RH2M', 'PRECTOTCORR']]
df.columns = ['date', 'temp', 'humidity', 'rain']

# Handle missing values
df = df.ffill()

# -------------------------------
# 2. FEATURE ENGINEERING
# -------------------------------

# Rolling features
df['temp_3d_avg'] = df['temp'].rolling(3).mean()
df['humidity_3d_avg'] = df['humidity'].rolling(3).mean()
df['rain_3d_sum'] = df['rain'].rolling(3).sum()

# Rain flag
df['rain_flag'] = (df['rain'] > 2).astype(int)

# Wet spell (consecutive rainy days)
df['wet_spell'] = df['rain_flag'] * (
    df['rain_flag'].groupby((df['rain_flag'] == 0).cumsum()).cumcount() + 1
)

# -------------------------------
# 🔥 NEW: Lag features (temporal learning)
# -------------------------------
df['temp_lag1'] = df['temp'].shift(1)
df['temp_lag2'] = df['temp'].shift(2)

df['humidity_lag1'] = df['humidity'].shift(1)
df['humidity_lag2'] = df['humidity'].shift(2)

df['rain_lag1'] = df['rain'].shift(1)
df['rain_lag2'] = df['rain'].shift(2)

# Optional seasonal feature
df['month'] = df['date'].dt.month
df['day_of_year'] = df['date'].dt.dayofyear

# 🔥 NEW: Weather Dynamics
df['temp_range'] = df['temp'].rolling(3).max() - df['temp'].rolling(3).min()
df['humidity_change'] = df['humidity'].diff()
df['rain_intensity'] = df['rain'] / (df['wet_spell'] + 1)

# Drop NaNs
df = df.dropna()

# -------------------------------
# 3. RED ROT RISK LABELING
# -------------------------------

def red_rot_risk(row):
    # Probabilistic scoring instead of hard thresholds
    score = 0
    
    # Humidity contribution (40%)
    score += min(row['humidity_3d_avg'] / 100, 1) * 0.4
    
    # Rain contribution (30%)
    score += min(row['rain_3d_sum'] / 30, 1) * 0.3
    
    # Temperature contribution (30%)
    score += (1 if 25 <= row['temp_3d_avg'] <= 30 else 0) * 0.3
    
    if score > 0.7:
        return 2
    elif score > 0.4:
        return 1
    else:
        return 0

df['risk'] = df.apply(red_rot_risk, axis=1)

# -------------------------------
# 4. FUTURE PREDICTION SETUP
# -------------------------------

# Predict multi-window (Level 1)
df['target_3d'] = df['risk'].shift(-3)
df['target_5d'] = df['risk'].shift(-5)

# Primary target for current model
df['target'] = df['target_3d']

df = df.dropna()

# -------------------------------
# 5. DATASET PREPARATION
# -------------------------------

features = [
    'temp', 'humidity', 'rain',
    'temp_3d_avg', 'humidity_3d_avg', 'rain_3d_sum',
    'wet_spell',
    'temp_lag1', 'temp_lag2',
    'humidity_lag1', 'humidity_lag2',
    'rain_lag1', 'rain_lag2',
    'month', 'day_of_year',
    'temp_range', 'humidity_change', 'rain_intensity'
]

X = df[features]
y = df['target']

# Time-based split
split = int(len(df) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -------------------------------
# 6. MODEL TRAINING
# -------------------------------

# Applying SMOTE for class balance (Level 2)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Advanced Model: XGBoost
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train_res, y_train_res)

# -------------------------------
# 7. EVALUATION
# -------------------------------

y_pred = model.predict(X_test)

print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

print("\n📊 Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))

# -------------------------------
# 8. CLASS DISTRIBUTION
# -------------------------------

print("\n⚖️ Class Distribution:\n")
print(df['target'].value_counts())

# -------------------------------
# 9. FEATURE IMPORTANCE
# -------------------------------

importance = model.feature_importances_

plt.figure()
plt.barh(features, importance)
plt.title("Feature Importance (Red Rot Prediction)")
plt.xlabel("Importance")
plt.show()

# -------------------------------
# 10. RISK VISUALIZATION
# -------------------------------

plt.figure(figsize=(12,5))
plt.plot(df['date'], df['risk'], label='Current Risk')
plt.plot(df['date'], df['target'], label='Future Risk (3 Days Ahead)')
plt.legend()
plt.title("Red Rot Risk Forecast")
plt.xlabel("Date")
plt.ylabel("Risk Level (0=Low, 1=Medium, 2=High)")
plt.show()

# -------------------------------
# 11. SAMPLE OUTPUT
# -------------------------------

print("\n🧪 Sample Predictions:\n")
print(df[['date', 'risk', 'target']].tail(10))
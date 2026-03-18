# Target Project Workflow: 11-Feature Diagnostic-Prognostic Pipeline

This document outlines the detailed sequence and feature space for the crop disease prediction system.

## Workflow Phases

### 1. Diagnosis Phase (LLM & Vision)
The system identifies key visual and descriptive attributes:
- **crop_type**: Species identified.
- **crop_age_days**: Age of the crop from user input/history.
- **disease_type**: Initial diagnosis from Vision-LLM.
- **severity**: Instantaneous severity of the observed disease (0.0 - 1.0).
- **infection_area**: Percentage of field/plant affected.

### 2. Environment Phase (Weather & Forecast)
The system fetches high-resolution weather data:
- **temperature**: Current temp (°C).
- **humidity**: Relative humidity (%).
- **rainfall**: Precipitation levels (mm).
- **wind_speed**: Wind velocity (km/h).
- **solar_radiation**: Energy flux (W/m²).
- **pressure**: Surface pressure (hPa).

### 3. Prognosis Phase (ML Risk Verification)
The XGBoost model processes the **11 features** to predict the future outlook:
- **future_severity**: Target variable predicting the disease progression.
- **risk_level**: Categorical risk (Low, Medium, High).

### 4. Recommendation Phase (LLM Insights)
Generates farmer-friendly, translated advice based on the prognostic outcome.

## Feature Mapping Summary
| Feature Name | Source | Type |
|--------------|--------|------|
| crop_type | LLM/User | Categorical |
| crop_age_days | User | Numerical |
| disease_type | LLM | Categorical |
| severity | LLM | Numerical |
| infection_area| LLM | Numerical |
| temperature | Weather | Numerical |
| humidity | Weather | Numerical |
| rainfall | Weather | Numerical |
| wind_speed | Weather | Numerical |
| solar_radiation| Weather | Numerical |
| pressure | Weather | Numerical |

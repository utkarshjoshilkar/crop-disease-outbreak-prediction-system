import httpx
import logging
import numpy as np

logger = logging.getLogger(__name__)

async def get_current_weather(lat: float, lon: float) -> dict:
    """
    Fetches the current weather and a 7-day forecast from the Open-Meteo API.
    """
    if lat is None or lon is None:
        return {"temperature": None, "humidity": None, "forecast_summary": "No GPS data"}
        
    # Requesting current and daily forecast data with expanded metrics
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,shortwave_radiation,surface_pressure"
        "&daily=temperature_2m_max,relative_humidity_2m_mean&timezone=auto"
    )
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            
            # 1. Current Stats
            current = data.get("current", {})
            temp = current.get("temperature_2m")
            humidity = current.get("relative_humidity_2m")
            precipitation = current.get("precipitation", 0.0)
            wind_speed = current.get("wind_speed_10m", 0.0)
            solar_radiation = current.get("shortwave_radiation", 0.0)
            pressure = current.get("surface_pressure", 1013.25)
            
            # 2. Daily Forecast Trends (Next 7 days)
            daily = data.get("daily", {})
            avg_humidity_forecast = np.mean(daily.get("relative_humidity_2m_mean", [humidity])) if daily.get("relative_humidity_2m_mean") else humidity
            avg_temp_max = np.mean(daily.get("temperature_2m_max", [temp])) if daily.get("temperature_2m_max") else temp
            
            # Outbreak Heuristic
            is_risk_increasing = avg_humidity_forecast > 75 and 20 < avg_temp_max < 30
            
            return {
                "temperature": temp,
                "humidity": humidity,
                "precipitation": precipitation,
                "wind_speed": wind_speed,
                "solar_radiation": solar_radiation,
                "pressure": pressure,
                "avg_humidity_forecast": float(avg_humidity_forecast),
                "avg_temp_max": float(avg_temp_max),
                "outbreak_trend": "Increasing Risk" if is_risk_increasing else "Stable",
                "forecast_summary": f"Next 7 days: Avg humidity {avg_humidity_forecast:.1f}%, Avg Max Temp {avg_temp_max:.1f}°C"
            }
    except Exception as e:
        logger.error(f"Failed to fetch weather data: {e}")
        return {"temperature": None, "humidity": None, "forecast_summary": "Weather service unavailable"}

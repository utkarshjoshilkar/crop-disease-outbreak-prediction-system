import httpx
import logging

logger = logging.getLogger(__name__)

async def get_current_weather(lat: float, lon: float) -> dict:
    """
    Fetches the current temperature and relative humidity from the Open-Meteo API.
    """
    if lat is None or lon is None:
        return {"temperature": None, "humidity": None}
        
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            
            # Open-Meteo returns 'current' object
            current = data.get("current", {})
            temp = current.get("temperature_2m")
            humidity = current.get("relative_humidity_2m")
            
            return {
                "temperature": temp,
                "humidity": humidity
            }
    except Exception as e:
        logger.error(f"Failed to fetch weather data: {e}")
        return {"temperature": None, "humidity": None}

import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_weather_forecast(lat=16.7, lon=74.69):
    """
    Fetches 7-day weather forecast from Open-Meteo API.
    Returns a DataFrame aggregated to daily with:
    - Temperature (Mean)
    - Humidity (Max) - More biological realism
    - Precipitation (Sum)
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation"],
        "timezone": "auto",
        "past_days": 7 # 7 days ensures stability for rolling/lag
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        hourly = data['hourly']
        df_hourly = pd.DataFrame({
            "time": pd.to_datetime(hourly['time']),
            "temp": hourly['temperature_2m'],
            "humidity": hourly['relative_humidity_2m'],
            "rain": hourly['precipitation']
        })
        
        # AGGREGATE TO DAILY
        df = df_hourly.groupby(df_hourly['time'].dt.date).agg({
            'temp': 'mean',
            'humidity': 'max',  # Catching peak infection windows
            'rain': 'sum'
        }).reset_index()
        df.rename(columns={'time': 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        
        return df
    except Exception as e:
        print(f"❌ Error fetching weather: {e}")
        return None

def get_forecast_display_data(df):
    """
    Subsets the data for display and basic risk mapping.
    """
    if df is None: return None
    # Only return upcoming 7 days for the dashboard
    today = datetime.now().date()
    return df[df['date'].dt.date >= today].copy()

if __name__ == "__main__":
    df = fetch_weather_forecast()
    print(df.tail(7))

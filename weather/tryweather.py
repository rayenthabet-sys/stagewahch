import pandas as pd
import requests
def get_weather_data(lat, lon, start_date, end_date):
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto"
    data = requests.get(url).json()
    
    df = pd.DataFrame(data['daily'])
    # Calculate GDD: (Max + Min)/2 - 10 (Base temperature for tomato)
    df['gdd'] = ((df['temperature_2m_max'] + df['temperature_2m_min']) / 2) - 10
    df['gdd'] = df['gdd'].clip(lower=0) # If it's cold, growth is 0, not negative
    
    print(f"☀️ Total Heat Accumulated (GDD): {df['gdd'].sum():.2f}")
    print(f"🌧️ Total Rain: {df['precipitation_sum'].sum():.2f} mm")
    return df

weather_df = get_weather_data(35.03, 9.48, "2024-04-15", "2024-07-15")
weather_df['temperature_2m_max'].plot(title="Daily Max Temp")
import requests

def get_soil_data(lat, lon):
    url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&property=clay&property=sand&depth=0-5cm&value=mean"
    response = requests.get(url).json()
    
    # Extracting Clay and Sand percentage
    clay = response['properties']['layers'][0]['depths'][0]['values']['mean'] / 10
    sand = response['properties']['layers'][1]['depths'][0]['values']['mean'] / 10
    
    print(f"📍 Soil at ({lat}, {lon}):")
    print(f"🧱 Clay: {clay}%")
    print(f"⏳ Sand: {sand}%")

get_soil_data(35.03, 9.48)
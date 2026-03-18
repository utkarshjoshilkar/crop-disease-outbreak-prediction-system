import requests
import json
import io

print("Testing the 11-feature /predict endpoint...")

# Create dummy image files
crop_img = io.BytesIO(b"fake crop image data")
crop_img.name = "test_crop.jpg"

soil_img = io.BytesIO(b"fake soil image data")
soil_img.name = "test_soil.png"

url = "http://127.0.0.1:8009/predict"

# 11-Feature payload (via Multipart Form)
files = {
    'crop_image': ('test_crop.jpg', crop_img, 'image/jpeg'),
    'soil_image': ('test_soil.png', soil_img, 'image/png')
}

data = {
    'latitude': 28.6139,
    'longitude': 77.2090,
    'description': "Lots of yellow spots on the leaves and wilting stems.",
    'native_language': "Hindi",
    'explicit_crop_type': "Soybean",
    'crop_age_days': 60
}

try:
    response = requests.post(url, files=files, data=data)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        res_json = response.json()
        print("Success!")
        print(f"Disease: {res_json.get('disease_type')}")
        print(f"Verified Disease: {res_json.get('verified_disease')}")
        print(f"Future Severity: {res_json.get('future_severity')}")
        print(f"Risk Level: {res_json.get('risk_level')}")
        print(f"Weather: {res_json.get('temperature')}°C, {res_json.get('humidity')}% Humidity")
        print(f"Rainfall: {res_json.get('rainfall')}mm")
        print("-" * 30)
        print("Recommendations:")
        print(res_json.get('llm_recommendations'))
    else:
        print("Error:")
        print(response.text)
except Exception as e:
    print(f"Failed to connect: {e}")
    print("Ensure the server is running with: uvicorn main:app --reload")

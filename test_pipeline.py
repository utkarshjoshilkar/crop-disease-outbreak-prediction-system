import requests
import json
import io

print("Testing the /predict endpoint with mock files...")

# Create dummy image files in memory
crop_img = io.BytesIO(b"fake crop image data")
crop_img.name = "test_crop.jpg"

soil_img = io.BytesIO(b"fake soil image data")
soil_img.name = "test_soil.png"

url = "http://127.0.0.1:8000/predict"

# Multipart payload
files = {
    'crop_image': ('test_crop.jpg', crop_img, 'image/jpeg'),
    'soil_image': ('test_soil.png', soil_img, 'image/png')
}

data = {
    'latitude': 28.6139,
    'longitude': 77.2090,
    'description': "Lots of yellow spots on the leaves.",
    'native_language': "Hindi"
}

try:
    response = requests.post(url, files=files, data=data)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Success!")
        print(json.dumps(response.json(), indent=2))
    else:
        print("Error:")
        print(response.text)
except Exception as e:
    print(f"Failed to connect: {e}")

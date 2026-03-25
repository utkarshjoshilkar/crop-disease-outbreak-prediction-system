import asyncio
import os
import sys

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from services.llm_service import extract_ml_features

async def test_vision_logic():
    image_path = r"c:\crop-disease-outbreak-prediction-system-main - Copy\crop-disease-outbreak-prediction-system-main\static\uploads\crop_20260318105206.jfif"
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    print(f"Testing vision logic with image: {image_path}")
    try:
        result = await extract_ml_features(
            description="Brown spots on leaves, what is this?",
            crop_image_path=image_path,
            soil_image_path=None,
            explicit_crop_type="Unknown"
        )
        print("\n--- VISION DIAGNOSIS SUCCESSFUL ---")
        print("Extracted Features:", result)
    except Exception as e:
        print(f"\n--- VISION DIAGNOSIS FAILED ---")
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_vision_logic())

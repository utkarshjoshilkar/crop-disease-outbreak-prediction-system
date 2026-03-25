import os

# LLM Configuration
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/api/generate")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
VISION_MODEL = os.getenv("VISION_MODEL", "llava:7b")

# Weather API Configuration (Placeholder - usually needs an API key)
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

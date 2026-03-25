import asyncio
import httpx
import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from config import LLM_BASE_URL, LLM_MODEL, VISION_MODEL

async def test_connection(model_name: str, label: str):
    print(f"Testing connection to {label} ({model_name}) at {LLM_BASE_URL}...")
    
    payload = {
        "model": model_name,
        "prompt": f"Hello {label}! Reply with '{label} is ready' if you can hear me.",
        "stream": False
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(LLM_BASE_URL, json=payload)
            response.raise_for_status()
            result = response.json()
            llm_response = result.get("response", "No response content")
            print(f"Success! {label} Response: {llm_response}\n")
    except httpx.ConnectError:
        print(f"Error: Could not connect to Ollama. Is it running?\n")
    except httpx.HTTPStatusError as e:
        print(f"Error: Received HTTP {e.response.status_code} from Ollama for {label}.\n")
        if e.response.status_code == 404:
            print(f"Model '{model_name}' might not be installed. Try running: ollama pull {model_name}\n")
    except Exception as e:
        print(f"An unexpected error occurred for {label}: {e}\n")

async def run_tests():
    await test_connection(LLM_MODEL, "Llama 3.1")
    await test_connection(VISION_MODEL, "Llava 7B")

if __name__ == "__main__":
    asyncio.run(run_tests())

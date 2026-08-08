import os
import time
from dataclasses import dataclass

import requests
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class GeminiConfig:
    api_key: str
    model: str
    api_version: str
    base_url: str
    max_retries: int


def get_gemini_config() -> GeminiConfig:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables. Please add it to your .env file.")

    model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    api_version = os.getenv("GEMINI_API_VERSION", "v1beta")
    base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com")
    max_retries = int(os.getenv("GEMINI_MAX_RETRIES", "5"))

    return GeminiConfig(
        api_key=api_key,
        model=model,
        api_version=api_version,
        base_url=base_url.rstrip("/"),
        max_retries=max_retries,
    )


def generate_gemini_text(prompt: str, model: str | None = None) -> str:
    config = get_gemini_config()
    model_name = model or config.model
    url = f"{config.base_url}/{config.api_version}/models/{model_name}:generateContent?key={config.api_key}"

    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt,
                    }
                ]
            }
        ]
    }

    headers = {"Content-Type": "application/json"}

    for attempt in range(config.max_retries):
        response = requests.post(url, headers=headers, json=data, timeout=120)

        if response.status_code == 200:
            response_data = response.json()
            return response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get(
                "text", "No response generated"
            )

        if response.status_code == 429:
            wait_time = 2 ** attempt
            print(f"Rate limit exceeded. Waiting for {wait_time} seconds before retrying...")
            time.sleep(wait_time)
            continue

        raise Exception(f"Error from Gemini API: {response.text}")

    raise Exception("Failed to retrieve content after multiple attempts due to rate limiting.")
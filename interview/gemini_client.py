# gemini_client.py

import os
import requests

# Make sure you’ve set:
#   export GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
GEMINI_KEY = "AIzaSyAEp_1C0w1I3sV2C-CyrJKxxz2J3kfOhc4"
if not GEMINI_KEY:
    raise RuntimeError("Please set the GOOGLE_API_KEY environment variable")

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/gemini-1.5-pro:generateContent"
)

def gemini_chat(prompt_text: str) -> str:
    """
    Send a single-text prompt to Gemini and return its response.
    Uses the v1beta generateContent endpoint with 'contents' + 'generationConfig'.
    """
    payload = {
        "contents": [
            { "parts": [ { "text": prompt_text } ] }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "topP": 0.8,
            "topK": 40,
            "maxOutputTokens": 256
        }
    }

    resp = requests.post(
        f"{GEMINI_URL}?key={GEMINI_KEY}",
        headers={ "Content-Type": "application/json" },
        json=payload
    )
    if resp.status_code != 200:
        try:
            details = resp.json()
        except ValueError:
            details = resp.text
        raise RuntimeError(f"Gemini API error {resp.status_code}: {details}")

    result = resp.json()
    return result["candidates"][0]["content"]["parts"][0]["text"].strip()

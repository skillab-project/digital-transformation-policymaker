
import requests
import json
from .config import settings

HEADERS = {
    "Authorization": f"Bearer {settings.api_token}",
    "Accept": "application/json"
}

def analyze_chunk(text: str, query: str, context: str, timeout: int):
    payload = {
        "model": settings.model_name,
        "messages": [{
            "role": "user",
            "content": f"{query}\n\nContext: {context}\n\n{text}"
        }],
        "temperature": settings.temperature,
        "seed": settings.seed,
        "format": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "technologies": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "domain": {"type": "string"},
                            "occupations": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "confidence": {"type": "number"}
                        },
                        "required": ["name", "description", "domain", "occupations", "confidence"]
                    }
                }
            },
            "required": ["technologies"]
        }
    }
    url = f"{settings.api_url}/api/chat/completions"
    resp = requests.post(url, headers=HEADERS, json=payload, timeout=timeout+10)
    resp.raise_for_status()
    outer = resp.json()
    content = outer["choices"][0]["message"]["content"]
    return json.loads(content)

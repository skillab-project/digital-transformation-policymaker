
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

def generate_json(text: str, query: str, context: str, timeout: int):
    payload = {
        "model": settings.model_name,
        "messages": [{
            "role": "user",
            "content": f"{query}\n\nContext: {context}\n\n{text}",
        }],
        "temperature": settings.temperature,
        "seed": settings.seed,
        "format": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "recommendations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "technology": {"type": "string"},
                            "actions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "area":       {"type": "string"},   # e.g., Training, HE Curricula, Funding, Standards, Incentives
                                        "action":     {"type": "string"},   # concrete step
                                        "rationale":  {"type": "string"},
                                        "stakeholders":{"type": "array", "items": {"type":"string"}},
                                        "timeframe":  {"type": "string"},   # short (0-6m), mid (6-24m), long (24m+)
                                        "KPIs":       {"type": "array", "items": {"type":"string"}},
                                        "risks":      {"type": "string"},
                                        "priority":   {"type": "string"}    # High/Medium/Low
                                    },
                                    "required": ["area","action","rationale","timeframe","priority"]
                                }
                            }
                        },
                        "required": ["technology","actions"]
                    }
                }
            },
            "required": ["recommendations"]
        }
    }
    url = f"{settings.api_url}/api/chat/completions"
    resp = requests.post(url, headers=HEADERS, json=payload, timeout=timeout + 10)
    resp.raise_for_status()
    outer = resp.json()
    content = outer["choices"][0]["message"]["content"]
    return json.loads(content)

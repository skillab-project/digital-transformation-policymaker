# -*- coding: utf-8 -*-
"""
Lightweight LLM client helpers.

Functions:
- analyze_chunk(text, query, context, timeout) -> dict
- generate_json(text, query, context, timeout) -> dict

Both call a JSON-enforcing chat endpoint and return parsed dicts.

Created on Thu Oct 23 13:33:15 2025

@author: tsoukj
"""

from __future__ import annotations


import json
import logging
import time
from typing import Any, Dict, Optional
import requests
from .config import settings

# ---------------------------------------------------------------------
# HTTP setup
# ---------------------------------------------------------------------
HEADERS: Dict[str, str] = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "User-Agent": "FutureTechTrends/1.0",
}
# Local backends (e.g. Ollama) need no token; only send one if configured
if settings.api_token:
    HEADERS["Authorization"] = f"Bearer {settings.api_token}"
# Allow project-specific extra headers without breaking defaults
if getattr(settings, "extra_headers", None):
    try:
        HEADERS.update(dict(settings.extra_headers))
    except Exception:  # best-effort only
        pass

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def analyze_chunk(text: str, query: str, context: str, timeout: int) -> Dict[str, Any]:
    """
    Analyze a document chunk and return extracted technologies as a dict.

    Returns:
        {
          "technologies": [
            {"name": ..., "description": ..., "domain": ..., "occupations": [...], "confidence": ...},
            ...
          ]
        }
    """
    payload = {
        "model": settings.model_name,
        "messages": [
            {
                "role": "user",
                "content": f"{query}\n\nContext: {context}\n\n{text}",
            }
        ],
        "temperature": settings.temperature,
        "seed": settings.seed,
        # JSON schema hint (some backends support this; harmless if ignored)
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
                            "occupations": {"type": "array", "items": {"type": "string"}},
                            "confidence": {"type": "number"},
                        },
                        "required": ["name", "description", "domain", "occupations", "confidence"],
                    },
                }
            },
            "required": ["technologies"],
        },
    }
    outer = _chat_json(url=_chat_url(), payload=_adapt_payload(payload), timeout=timeout + 10)
    return outer

def generate_json(text: str, query: str, context: str, timeout: int) -> Dict[str, Any]:
    """
    Generate policy recommendations JSON for an 'emerging' technology block.

    Returns:
        {
          "recommendations": [
            {"technology": "...", "actions": [ {area, action, rationale, timeframe, priority, ...}, ... ]},
            ...
          ]
        }
    """
    payload = {
        "model": settings.model_name,
        "messages": [
            {
                "role": "user",
                "content": f"{query}\n\nContext: {context}\n\n{text}",
            }
        ],
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
                                        "area": {"type": "string"},
                                        "action": {"type": "string"},
                                        "rationale": {"type": "string"},
                                        "stakeholders": {"type": "array", "items": {"type": "string"}},
                                        "timeframe": {"type": "string"},
                                        "KPIs": {"type": "array", "items": {"type": "string"}},
                                        "risks": {"type": "string"},
                                        "priority": {"type": "string"},
                                    },
                                    "required": ["area", "action", "rationale", "timeframe", "priority"],
                                },
                            },
                        },
                        "required": ["technology", "actions"],
                    },
                }
            },
            "required": ["recommendations"],
        },
    }
    outer = _chat_json(url=_chat_url(), payload=_adapt_payload(payload), timeout=timeout + 10)
    return outer

# ---------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------
def _chat_url() -> str:
    """Full chat-completions URL for the configured backend."""
    return f"{settings.api_url.rstrip('/')}{settings.chat_path}"

def _adapt_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Translate the JSON-schema hint to whatever the configured backend understands.

    - OpenAI-compatible endpoints (e.g. Ollama's /v1) reject/ignore the top-level
      `format` key; they use `response_format` instead.
    - OpenWebUI/Ollama-native endpoints keep `format` as-is.
    """
    openai_style = not settings.chat_path.startswith("/api/")
    if not openai_style:
        return payload

    adapted = dict(payload)
    schema = adapted.pop("format", None)
    if schema is not None:
        adapted["response_format"] = {"type": "json_object"}
    return adapted

def _chat_json(url: str, payload: Dict[str, Any], timeout: int, retries: int = 2) -> Dict[str, Any]:
    """
    POST a chat completion request and parse a JSON object from the first choice's message.

    - Retries transient failures (HTTP >= 500, JSON parse errors) up to `retries` times
    - Accepts both strict JSON responses and "markdown-fenced" JSON
    - Raises HTTPError or ValueError with context on persistent failure
    """
    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, headers=HEADERS, json=payload, timeout=timeout)
            resp.raise_for_status()

            data = resp.json()
            content = _extract_choice_content(data)
            parsed = _parse_json_relaxed(content)

            if not isinstance(parsed, dict):
                raise ValueError("Model returned non-object JSON.")

            return parsed

        except requests.HTTPError as e:
            # Do not retry 4xx (client errors); retry 5xx
            status = getattr(e.response, "status_code", None)
            if status and 400 <= status < 500:
                raise
            last_err = e

        except (ValueError, json.JSONDecodeError) as e:
            # JSON parse issues — retry could help if model was flaky
            last_err = e

        except requests.RequestException as e:
            # Connectivity/timeouts
            last_err = e

        # Backoff before retry (simple linear backoff)
        if attempt < retries:
            sleep_s = 0.75 * (attempt + 1)
            time.sleep(sleep_s)

    # If we get here, retries exhausted
    raise RuntimeError(f"LLM request failed after {retries + 1} attempt(s): {last_err}")

def _extract_choice_content(outer: Dict[str, Any]) -> str:
    """
    Extract the assistant message 'content' from a /chat/completions payload.
    Throws ValueError if the expected structure is missing.
    """
    try:
        choices = outer["choices"]
        if not choices:
            raise KeyError("choices is empty")
        msg = choices[0]["message"]
        content = msg.get("content", "")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Empty content from model.")
        return content
    except (KeyError, TypeError) as e:
        raise ValueError(f"Unexpected response shape from LLM: {e}; outer keys={list(outer.keys())}") from e

def _parse_json_relaxed(text: str) -> Any:
    """
    Parse JSON with a few conveniences:
    - Strips Markdown code fences (```json ... ```)
    - If raw parsing fails, attempts a best-effort brace slice
    """
    s = text.strip()

    # Strip triple-backtick fences if present
    if s.startswith("```"):
        # ```json\n...\n```
        s = _strip_code_fences(s)

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Try best-effort: take substring between first '{' and last '}'.
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = s[start : end + 1]
            return json.loads(candidate)
        raise

def _strip_code_fences(s: str) -> str:
    """
    Remove leading and trailing triple-backtick code fences from a string.
    Keeps inner content intact.
    """
    lines = s.splitlines()
    if not lines:
        return s
    # Drop first fence
    if lines[0].startswith("```"):
        lines = lines[1:]
    # Drop trailing fence if present
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)

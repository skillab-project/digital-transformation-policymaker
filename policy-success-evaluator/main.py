# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 13:33:15 2025

@author: tsoukj
"""

# FastAPI service for KPI → LLM → Policy Recommendations (US #19)
# ---------------------------------------------------------------
# • POST /kpi/recommendations : returns structured recommendations
# • Uses env-configured LLM server compatible with /api/chat/completions
#
#
# Run:
#   uvicorn main:app --reload --port 8000

from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError, validator
from dotenv import load_dotenv

# =========================
# Config (env-based)
# =========================
load_dotenv()

class Settings(BaseModel):
    api_url: str = os.getenv("API_URL", "<REPLACE_ME>")
    api_token: str = os.getenv("API_TOKEN", "<REPLACE_ME>")
    model_name: str = os.getenv("MODEL_NAME", "mistral:latest")
    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))
    seed: int = int(os.getenv("SEED", "42"))
    timeout: int = int(os.getenv("TIMEOUT", "60"))

settings = Settings()

HEADERS = {
    "Authorization": f"Bearer {settings.api_token}",
    "Accept": "application/json",
    "Content-Type": "application/json",
    "User-Agent": "SKILLAB-KPI-Recs/1.0",
}

# =========================
# Schemas
# =========================
class TimePoint(BaseModel):
    period: str  # "YYYY-Qx" or "YYYY-MM"
    value: float

class KPI(BaseModel):
    id: str
    name: str
    unit: str
    direction: str  # "higher_is_better" | "lower_is_better"
    current_value: float
    target_value: float
    target_deadline: str  # "YYYY-Qx" or "YYYY-MM"
    last_updated: Optional[str] = None
    time_series: Optional[List[TimePoint]] = None

    @validator("direction")
    def _direction_ok(cls, v: str):
        v = v.strip().lower()
        if v not in ("higher_is_better", "lower_is_better"):
            raise ValueError("direction must be 'higher_is_better' or 'lower_is_better'")
        return v

class Scope(BaseModel):
    sector: str
    region: str
    policy: str

class RecsRequest(BaseModel):
    kpis: List[KPI]
    scope: Scope

class RecommendationItem(BaseModel):
    lever_type: str
    title: str
    mechanism: str
    rational: str
    expected_impact: str  # Low/Medium/High
    time_to_effect: str   # Short/Medium/Long
    risks_tradeoffs: str
    prerequisites: List[str]

class RecsResponse(BaseModel):
    kpi_id: str
    trend_analysis: Optional[str] = None
    recommendations: List[RecommendationItem]

# =========================
# Trend utilities
# =========================
def _parse_period(period: str) -> Tuple[int, int, str]:
    s = period.strip()
    if "-Q" in s.upper():
        y, q = s.upper().split("-Q")
        return int(y), int(q), "Q"
    else:
        y, m = s.split("-")
        return int(y), int(m), "M"

def _to_ordinal(period: str) -> Tuple[int, str]:
    y, idx, f = _parse_period(period)
    return (y * (4 if f == "Q" else 12) + (idx - 1), f)

def _from_ordinal(ordinal: int, freq: str) -> str:
    if freq == "Q":
        y, r = divmod(ordinal, 4)
        return f"{y}-Q{r+1}"
    y, r = divmod(ordinal, 12)
    return f"{y}-{r+1:02d}"

def _linreg_slope(x: List[float], y: List[float]) -> float:
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = sum((xi - mx) ** 2 for xi in x)
    return 0.0 if den == 0 else num / den

def describe_trend(kpi: KPI) -> Dict[str, Any]:
    ts = kpi.time_series or []
    if len(ts) < 3:
        return {"trend_summary": "Insufficient history to assess trend reliably."}
    points = sorted(ts, key=lambda p: _to_ordinal(p.period)[0])
    ords = [_to_ordinal(p.period)[0] for p in points]
    freq = _to_ordinal(points[-1].period)[1]
    x = [o - ords[0] for o in ords]
    y = [p.value for p in points]
    slope = _linreg_slope(x, y)  # value per period
    eps = 1e-9
    if kpi.direction == "higher_is_better":
        signed_gap = kpi.target_value - kpi.current_value
        improving = slope > eps
    else:
        signed_gap = kpi.current_value - kpi.target_value
        improving = slope < -eps
    abs_gap = abs(signed_gap)
    if abs(slope) > eps and ((kpi.direction == "higher_is_better" and slope > 0) or (kpi.direction == "lower_is_better" and slope < 0)):
        projected = math.ceil(abs_gap / abs(slope))
    else:
        projected = math.inf
    # periods until deadline
    last_period = points[-1].period
    until_deadline = _to_ordinal(kpi.target_deadline)[0] - _to_ordinal(last_period)[0]
    on_track = projected != math.inf and projected <= max(0, until_deadline)
    per = "per quarter" if freq == "Q" else "per month"
    pace = "increasing" if slope > eps else ("decreasing" if slope < -eps else "flat")
    eta = (f"~{projected} {'quarters' if freq=='Q' else 'months'}") if projected != math.inf else "not reachable at current pace"
    meets = "on track" if on_track else "off track"
    summary = f"{pace.capitalize()} at {slope:+.2f} {per}. On current pace, target met in {eta}. This is {meets} relative to deadline {kpi.target_deadline}."
    return {
        "slope_per_period": slope,
        "projected_periods_to_target": None if projected == math.inf else projected,
        "periods_until_deadline": until_deadline,
        "on_track": on_track,
        "trend_summary": summary,
    }

# =========================
# LLM client (JSON-enforced)
# =========================
def _strip_code_fences(s: str) -> str:
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\n?", "", s, count=1, flags=re.MULTILINE)
        s = re.sub(r"\n?```$", "", s, count=1, flags=re.MULTILINE)
    return s

def _chat_json(payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    url = f"{settings.api_url}/api/chat/completions"
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=HEADERS, json=payload, timeout=timeout)
            resp.raise_for_status()
            outer = resp.json()
            content = outer["choices"][0]["message"]["content"]
            content = _strip_code_fences(content or "")
            return json.loads(content)
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(0.75 * (attempt + 1))
    raise RuntimeError("Unreachable")

def _split_prereqs(x: Any) -> List[str]:
    if isinstance(x, list):
        return [p.strip() for p in x if str(p).strip()]
    if isinstance(x, str):
        parts = [p.strip() for p in re.split(r"[;,]", x) if p.strip()]
        return parts or ["Define prerequisites"]
    return ["Define prerequisites"]

def _norm_tte(val: str, allowed: List[str]) -> str:
    v = (val or "").strip().title()
    # fix variants like "Short to Medium"
    if "Short" in v and "Medium" in v:
        return "Medium" if "Medium" in allowed else allowed[0]
    return v if v in allowed else allowed[0]

def _enforce_on_track_policy(on_track: Optional[bool], recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if on_track is not True:
        return recs

    allowed_levers = {"Monitoring", "Consolidation", "Advisory Support", "Evaluation", "Data Collection"}
    allowed_impacts = {"Low", "Medium"}
    allowed_tte = ["Short", "Medium"]

    replacements = [
        # (keyword, new_title, mech_suffix)
        ("training", "Lightweight Skills Monitoring & Nudges",
         "Monitor participation and outcomes of existing training; send targeted nudges to sustain uptake without new subsidies."),
        ("collaborate", "Provider Performance Monitoring & Light Brokerage",
         "Track provider delivery quality; publish light scorecards; broker matches only where bottlenecks appear."),
        ("policy", "Regulatory Impact Monitoring",
         "Assess unintended burdens/benefits quarterly; adjust guidance if early warnings trigger."),
    ]

    fixed: List[Dict[str, Any]] = []
    for r in recs:
        item = dict(r)

        # Enforce lever type
        if item.get("lever_type") not in allowed_levers:
            item["lever_type"] = "Monitoring"

        # Rewrite titles/mechanisms toward consolidation if they look like accelerators
        title = (item.get("title") or "").strip()
        mech  = (item.get("mechanism") or "").strip()

        low = (title + " " + mech).lower()
        adjusted = False
        for key, new_title, tail in replacements:
            if key in low:
                item["title"] = new_title
                item["mechanism"] = f"Establish light-touch tracking with early-warning thresholds. {tail}"
                adjusted = True
                break
        if not adjusted:
            # Generic consolidation phrasing if nothing matched
            if item.get("lever_type") == "Monitoring":
                item["title"] = title or "Lightweight Monitoring & Early-Warning"
                item["mechanism"] = mech or "Quarterly checks with threshold-based alerts; minimal reporting load."

        # Clamp impact
        imp = (item.get("expected_impact") or "").title()
        item["expected_impact"] = imp if imp in allowed_impacts else "Medium"

        # Clamp time_to_effect
        item["time_to_effect"] = _norm_tte(item.get("time_to_effect", ""), allowed_tte)

        # Normalize prerequisites
        item["prerequisites"] = _split_prereqs(item.get("prerequisites"))

        # Ensure rational exists
        if not item.get("rational"):
            item["rational"] = "KPI is on track; consolidation minimizes cost while preserving trajectory and detecting regressions early."

        fixed.append(item)

    if not fixed:
        fixed = [{
            "lever_type": "Monitoring",
            "title": "Lightweight Monitoring & Early-Warning",
            "mechanism": "Implement quarterly checks with threshold alerts and minimal reporting load.",
            "rational": "KPI is on track; focus on maintaining trajectory and quick response to deviations.",
            "expected_impact": "Medium",
            "time_to_effect": "Short",
            "risks_tradeoffs": "Risk of complacency; keep alert thresholds meaningful.",
            "prerequisites": ["Define thresholds", "Assign monitoring owner"]
        }]
    return fixed

def call_llm_for_recommendations(kpi: KPI, scope: Scope, trend_summary: Optional[str], on_track: Optional[bool] = None) -> Dict[str, Any]:
    gap = (kpi.target_value - kpi.current_value) if kpi.direction == "higher_is_better" else (kpi.current_value - kpi.target_value)

    # === conditional lever sets based on track status ===
    if on_track is True:
        lever_enum = ["Monitoring", "Consolidation", "Advisory Support", "Evaluation", "Data Collection"]
        impact_enum = ["Low", "Medium"]
        tte_enum = ["Short", "Medium"]                      # <-- restrict time_to_effect when on-track
        mode_line = "Status: ON_TRACK — focus on consolidation/monitoring; avoid cost-intensive levers (no Grants/Tax Incentives)."
    else:
        lever_enum = ["Grants", "Tax Incentives", "Training", "Regulation", "Public Procurement", "Partnerships", "Advisory Support"]
        impact_enum = ["Low", "Medium", "High"]
        tte_enum = ["Short", "Medium", "Long"]              # <-- full range when not on-track
        mode_line = "Status: NOT_ON_TRACK — prioritize accelerators; cost-effective but impactful levers."

    system_block = (
        "You are an expert policy analyst supporting the SKILLAB platform.\n"
        "Use historical KPI trends and current gaps to recommend actions that can realistically "
        "accelerate progress toward the target within the stated deadline."
        "Your recommendations whould allign with EU priorities (digital, green, skills, competitiveness)"
        "Be specific to the given sector/region/policy."
    )

    user_block = (
        "Here is the KPI information:\n\n"
        f"- KPI ID: {kpi.id}\n"
        f"- KPI Name: {kpi.name}\n"
        f"- Unit: {kpi.unit}\n"
        f"- Direction: {kpi.direction}  # higher_is_better | lower_is_better\n"
        f"- Current Value: {kpi.current_value}\n"
        f"- Target Value: {kpi.target_value}\n"
        f"- Gap (directional): {gap}  # target-current if higher_is_better; current-target if lower_is_better\n"
        f"- Target Deadline: {kpi.target_deadline}\n"
        f"- Sector: {scope.sector}\n"
        f"- Region: {scope.region}\n"
        f"- Policy: {scope.policy}\n"
        f"- {mode_line}\n"
    )

    if trend_summary:
        # include optional trend info and timeseries data
        ts_json = json.dumps([tp.dict() for tp in (kpi.time_series or [])], ensure_ascii=False)
        user_block += f"- Time Series: {ts_json}\n"
        user_block += f"- Trend Summary: {trend_summary}\n"

    # ----- Rules / instructions block -----
    rules = (
        "Task:\n"
        "1. Assess the KPI trajectory:\n"
        "   - Is the KPI improving, stagnating, or declining?\n"
        "   - Is the current trend sufficient to reach the target by the deadline?\n"
        "2. Propose 3–5 interventions.\n"
        "3. For each recommendation, include exactly these fields:\n"
        "   - lever_type\n"
        "   - title\n"
        "   - mechanism\n"
        "   - rational (a brief 1–2 sentence justification linking the mechanism to closing this KPI gap)\n"
        "   - expected_impact (Low | Medium | High)\n"
        "   - time_to_effect (Short | Medium | Long)\n"
        "   - risks_tradeoffs\n"
        "   - prerequisites (array of 1–3 short items)\n"
        "4. If status is ON_TRACK, DO NOT use cost-intensive levers like Grants or Tax Incentives; focus on consolidation/monitoring.\n\n"
        "Output JSON format (return ONLY valid JSON, no prose, no markdown):\n\n"
        "{\n"
        f'  "kpi_id": "{kpi.id}",\n'
        '  "trend_analysis": "…",\n'
        '  "recommendations": [\n'
        "    {\n"
        '      "lever_type": "…",\n'
        '      "title": "…",\n'
        '      "mechanism": "…",\n'
        '      "rational": "…"\n'
        '      "expected_impact": "Low|Medium|High",\n'
        '      "time_to_effect": "Short|Medium|Long",\n'
        '      "risks_tradeoffs": "…",\n'
        '      "prerequisites": ["…", "…"]\n'
        "    }\n"
        "  ]\n"
        "}"
    )

    payload = {
        "model": settings.model_name,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"System instructions:\n{system_block}\n\n"
                    f"User data:\n{user_block}\n\n"
                    f"Task and Output Rules:\n{rules}"
                ),
            }
        ],
        "temperature": settings.temperature,
        "seed": settings.seed,
        "format": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "kpi_id": {"type": "string"},
                "trend_analysis": {"type": "string"},
                "recommendations": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "lever_type": {"type": "string", "enum": lever_enum},
                            "title": {"type": "string"},
                            "mechanism": {"type": "string"},
                            "rational": {"type": "string"},
                            "expected_impact": {"type": "string", "enum": impact_enum},
                            "time_to_effect": {"type": "string", "enum": tte_enum},
                            "risks_tradeoffs": {"type": "string"},
                            "prerequisites": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 3,
                                "items": {"type": "string"}
                            }
                        },
                        "required": [
                            "lever_type",
                            "title",
                            "mechanism",
                            "rational",
                            "expected_impact",
                            "time_to_effect",
                            "risks_tradeoffs",
                            "prerequisites"
                        ],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["kpi_id", "recommendations"],
            "additionalProperties": False
        },
    }

    return _chat_json(payload, timeout=settings.timeout + 10)


# =========================
# FastAPI app
# =========================
app = FastAPI(title="SKILLAB KPI Recommendation Service", version="1.0.0")

@app.post("/kpi/recommendations", response_model=List[RecsResponse])
def kpi_recommendations(req: RecsRequest):
    results: List[RecsResponse] = []

    for kpi in req.kpis:
        # Compute trend
        trend = describe_trend(kpi) if kpi.time_series else {"trend_summary": None, "on_track": None}
        trend_summary = trend.get("trend_summary")
        on_track = trend.get("on_track")

        try:
            raw = call_llm_for_recommendations(kpi, req.scope, trend_summary, on_track=on_track)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"LLM call failed for KPI '{kpi.id}': {e}")

        try:
            recs = raw.get("recommendations", [])
            recs = _enforce_on_track_policy(on_track, recs)
            
            payload = {
                "kpi_id": raw.get("kpi_id", kpi.id),
                "trend_analysis": trend_summary,
                "recommendations": recs,
            }
            results.append(RecsResponse(**payload))
        except ValidationError as ve:
            raise HTTPException(
                status_code=500,
                detail=f"Invalid LLM JSON schema for KPI '{kpi.id}': {ve}"
            )

    return results

@app.get("/health")
def health():
    return {"status": "ok", "model": settings.model_name}

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 13:33:15 2025

@author: tsoukj
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import requests
from dateutil import parser
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError, validator
from dotenv import load_dotenv

# =========================
# Config (env-based)
# =========================
load_dotenv()

class Settings(BaseModel):
    api_url: str = os.getenv("API_URL", "<REPLACE_ME>")
    api_token: str = os.getenv("API_TOKEN", "<REPLACE_ME>")
    model: str = os.getenv("MODEL_NAME", "mistral:latest")
    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))
    seed: int = int(os.getenv("SEED", "42"))
    timeout: int = int(os.getenv("TIMEOUT", "60"))
    policy_base: str = os.getenv("POLICY_BASE", "https://portal.skillab-project.eu/policy")

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
    description: str

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

class PolicyRequest(BaseModel):
    policy_name: str
    kpi_name: Optional[str] = None

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
    
    # Signed gap
    if kpi.direction == "higher_is_better":
        signed_gap = kpi.target_value - kpi.current_value
        improving = slope > eps
    else:
        signed_gap = kpi.current_value - kpi.target_value
        improving = slope < -eps
    
    abs_gap = abs(signed_gap)
    
    # Projected periods to reach goal
    if abs(slope) > eps and improving:
        projected = math.ceil(abs_gap / abs(slope))
    else:
        projected = math.inf
    
    # Time to deadline
    last_period = points[-1].period
    until_deadline = _to_ordinal(kpi.target_deadline)[0] - _to_ordinal(last_period)[0]
    
    # Whether KPI will meet target before deadline
    on_track = projected != math.inf and projected <= max(0, until_deadline)
    
    # Human-readable summary
    per = "per quarter" if freq == "Q" else "per month"
    pace = "increasing" if slope > eps else ("decreasing" if slope < -eps else "flat")
    eta = (f"~{projected} {'quarters' if freq=='Q' else 'months'}") if projected != math.inf else "not reachable at current pace"
    meets = "on track" if on_track else "off track"
    
    summary = f"{pace.capitalize()} at {slope:+.2f} {per}. On current pace, target met in {eta}. This is {meets} relative to deadline {kpi.target_deadline}."
    
    return {
        "slope_per_period": slope,
        "improving": improving,
        "projected_periods_to_target": None if projected == math.inf else projected,
        "periods_until_deadline": until_deadline,
        "on_track": on_track,
        "trend_summary": summary,
    }

# =================
# Fetch Policy data
# =================
def fetch_policy_metadata(policy_name: str):
    url = f"{settings.policy_base}/policy"
    res = requests.get(url, params={"name": policy_name})

    if res.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch policy metadata: {res.status_code}"
        )

    data = res.json()
    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"No policy found with name '{policy_name}'"
        )

    return data

def fetch_kpi_timeseries(kpi_name: str) -> List[TimePoint]:
    url = f"{settings.policy_base}/report/kpi"
    res = requests.get(url, params={"kpiName": kpi_name})

    if res.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch measurements for KPI '{kpi_name}': {res.status_code}"
        )

    entries = res.json()
    timeseries = []

    for entry in entries:
        dt = parser.parse(entry["date"])
        quarter = (dt.month - 1) // 3 + 1
        period = f"{dt.year}-Q{quarter}"

        timeseries.append(TimePoint(
            period=period,
            value=entry["value"]
        ))

    timeseries = sorted(timeseries, key=lambda x: x.period)
    return timeseries

def build_kpi_from_policy_meta(kpi_meta, timeseries: List[TimePoint]) -> KPI:
    # Last measurement = current value
    current_value = timeseries[-1].value if timeseries else 0.0

    # Convert `31/12/2026` to `2026-Q4`
    try:
        d, m, y = kpi_meta["targetTime"].split("/")
        quarter = (int(m) - 1) // 3 + 1
        target_deadline = f"{y}-Q{quarter}"
    except:
        target_deadline = "2030-Q4"

    return KPI(
        id=str(kpi_meta["id"]),
        name=kpi_meta["name"],
        unit="percentage",               # you may refine later
        direction="higher_is_better",    # default for digital KPIs
        current_value=current_value,
        target_value=kpi_meta["targetValue"],
        target_deadline=target_deadline,
        time_series=timeseries
    )

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

def call_llm_for_recommendations(kpi: KPI, scope: Scope, trend_summary: Optional[str], on_track: Optional[bool] = None) -> Dict[str, Any]:
    gap = (kpi.target_value - kpi.current_value) if kpi.direction == "higher_is_better" else (kpi.current_value - kpi.target_value)

    # === conditional lever sets based on track status ===
    lever_enum_on_track = ["Monitoring", "Consolidation", "Advisory Support", "Evaluation", "Data Collection"]
    lever_enum_off_track = ["Grants", "Tax Incentives", "Training", "Regulation", "Public Procurement", "Partnerships", "Advisory Support"]
    if on_track is True:
        lever_enum = lever_enum_on_track
        impact_enum = ["Low", "Medium"]
        tte_enum = ["Short", "Medium"]                      # <-- restrict time_to_effect when on-track
        mode_line = "Status: ON_TRACK — focus on consolidation/monitoring; avoid cost-intensive levers (no Grants/Tax Incentives)."
    else:
        lever_enum = lever_enum_off_track
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
        f"- Policy Name: {scope.policy}\n"
        f"- Policy Description: {scope.description}\n"
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
        f'2. Propose 5-{len(lever_enum_off_track)} interventions from the categories: {lever_enum_off_track}.\n'
        "3. For each recommendation, include exactly these fields:\n"
        "   - lever_type\n"
        "   - title\n"
        "   - mechanism (a brief 2–3 sentence description of the mechanism)\n"
        "   - rational (a brief 1–2 sentence justification linking the mechanism to closing this KPI gap)\n"
        "   - expected_impact (Low | Medium | High)\n"
        "   - time_to_effect (Short | Medium | Long)\n"
        "   - risks_tradeoffs\n"
        "   - prerequisites (array of 1–3 short items)\n"
        f'4. If status is ON_TRACK, DO NOT use cost-intensive levers like Grants or Tax Incentives; focus on categories: {lever_enum_on_track}.\n\n'
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
        "model": settings.model,
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
                    "minItems": len(lever_enum),
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
            payload = {
                "kpi_id": raw.get("kpi_id", kpi.id),
                "trend_analysis": trend_summary,
                "recommendations": raw.get("recommendations", []),
            }
            results.append(RecsResponse(**payload))
        except ValidationError as ve:
            raise HTTPException(
                status_code=500,
                detail=f"Invalid LLM JSON schema for KPI '{kpi.id}': {ve}"
            )

    return results

@app.post("/policy/recommendations", response_model=List[RecsResponse])
def policy_recommendations(req: PolicyRequest):
    # Step 1 — Get policy metadata
    policy = fetch_policy_metadata(req.policy_name)

    results: List[RecsResponse] = []

    target_kpi_name = (req.kpi_name or "").strip()

    # Step 2 — Loop through all KPIs in the policy
    for kpi_meta in policy.get("kpiList", []):
        kpi_name = kpi_meta["name"]

        # If user asked for a specific KPI, skip others
        if target_kpi_name and kpi_name != target_kpi_name:
            continue

        # Step 3 — Fetch KPI time-series from SKILLAB Portal API
        ts = fetch_kpi_timeseries(kpi_name)

        # Step 4 — Convert to internal KPI object
        kpi_obj = build_kpi_from_policy_meta(kpi_meta, ts)

        # Step 5 — Trend analysis
        trend = describe_trend(kpi_obj) if ts else {"trend_summary": None, "on_track": None}

        # Step 6 — LLM recommendations
        safe_desc = (policy.get("description") or "").replace("\n", " ").replace("\r", " ")
        
        raw = call_llm_for_recommendations(
            kpi=kpi_obj,
            scope=Scope(
                sector=policy.get("sector", "Unknown"),
                region=policy.get("region", "Unknown"),
                policy=policy.get("name", req.policy_name),
                description=safe_desc
            ),
            trend_summary=trend.get("trend_summary"),
            on_track=trend.get("on_track")
        )

        # Step 7 — Build response
        results.append(RecsResponse(
            kpi_id=kpi_obj.id,
            trend_analysis=trend.get("trend_summary"),
            recommendations=raw.get("recommendations", [])
        ))

    if target_kpi_name and not results:
        # user asked for a specific KPI but it wasn't found in policy.kpiList
        raise HTTPException(
            status_code=404,
            detail=f"KPI '{target_kpi_name}' not found in policy '{req.policy_name}'"
        )

    return results

@app.get("/health")
def health():
    return {"status": "ok", "model": settings.model}

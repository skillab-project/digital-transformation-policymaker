# -*- coding: utf-8 -*-
"""
Pydantic models for the Future Tech Trends Analyzer.

Created on Thu Oct 23 13:33:15 2025

@author: tsoukj
"""

from __future__ import annotations

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

# ---------------------------------------------------------------------
# Core domain
# ---------------------------------------------------------------------
class Technology(BaseModel):
    """Minimal technology object produced by the analyzer."""
    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "name": "Edge AI in Industrial IoT",
                "description": "On-device inference for predictive maintenance.",
                "domain": "AI/IoT",
                "occupations": ["Industrial Engineer", "Data Scientist"],
                "confidence": 0.82,
            }
        }
    )

    name: str = Field(..., description="Canonical technology name")
    description: str = Field("", description="Short description or context")
    domain: str = Field("", description="High-level domain/category")
    occupations: Optional[List[str]] = Field(
        default=None, description="Related occupations (if any)"
    )
    confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Extractor confidence in [0,1]"
    )

class AnalysisResult(BaseModel):
    """Structured output of PDF analysis."""
    technologies: List[Technology]

class AnalyzeRequest(BaseModel):
    """Optional body for analysis endpoints when a text query is used."""
    query: str = Field(..., min_length=1)

# ---------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------
class JobStatus(BaseModel):
    """Status envelope for background analysis jobs."""
    job_id: str
    status: Literal["pending", "queued", "running", "done", "error"]
    message: Optional[str] = None
    result_path: Optional[str] = None
    user_id: Optional[str] = None
    source_job_id: Optional[str] = None
    type: Optional[str] = None

# ---------------------------------------------------------------------
# ESCO mapping
# ---------------------------------------------------------------------
class ESCOItem(BaseModel):
    """One ESCO match with a similarity score (0..1)."""
    label: str
    score: float = Field(..., ge=0.0, le=1.0)

class ESCOMapItem(BaseModel):
    """Mapping result for a single technology to one ESCO side."""
    technology: str
    matches: List[ESCOItem]

class ESCOMapBoth(BaseModel):
    """Combined mapping: occupations and skills lists."""
    occupations: List[ESCOMapItem]
    skills: List[ESCOMapItem]

class ESCOMapRequest(BaseModel):
    """
    Request to map technologies to ESCO.

    Supply either:
      - `technologies`: inline list, or
      - `job_id`: to load technologies from a previous analysis
    """
    job_id: Optional[str] = Field(
        default=None, description="Existing analysis job id to source technologies from"
    )
    technologies: Optional[List[Technology]] = Field(
        default=None, description="Inline technology list to map"
    )
    top_n: int = Field(5, ge=1, le=50, description="Max matches per technology")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Similarity threshold")
    target: Literal["occupations", "skills", "both"] = Field(
        "both", description="ESCO side(s) to map against"
    )

# ---------------------------------------------------------------------
# Policy recommendations
# ---------------------------------------------------------------------
class ActionItem(BaseModel):
    """One concrete action in a recommendation."""
    area: str = Field(..., description="Policy area (e.g., Training/Reskilling)")
    action: str = Field(..., description="Concrete step")
    rationale: str = Field(..., description="Why this matters")
    stakeholders: Optional[List[str]] = Field(default=None, description="Who should act")
    timeframe: str = Field(..., description="short | medium | long")
    KPIs: Optional[List[str]] = Field(default=None, description="Suggested KPIs")
    risks: Optional[str] = Field(default=None, description="Potential risks")
    priority: str = Field(..., description="High/Medium/Low")

class RecommendationItem(BaseModel):
    """Recommendations for one technology."""
    technology: str
    actions: List[ActionItem]

class PolicyReq(BaseModel):
    """
    Request to generate policy recommendations.

    Supply either:
      - `technologies`: inline list, or
      - `job_id`: to load technologies from a previous analysis
    """
    job_id: Optional[str] = Field(
        default=None, description="Existing analysis job id to source technologies from"
    )
    user_id: Optional[str] = Field(
        default=None, description="Optional user identifier to associate with this policy job"
    )
    # Keep Dict to tolerate loose upstream payloads from LLM extraction
    technologies: Optional[List[Dict]] = Field(
        default=None, description="Inline technology list (dicts tolerated)"
    )
    target: Literal["skills", "occupations", "both"] = Field(
        "both", description="Which ESCO side(s) to consider as evidence"
    )
    similarity_threshold: float = Field(0.5, ge=0.0, le=1.0)
    max_actions_per_tech: int = Field(4, ge=1, le=10)
    llm_model: Optional[str] = Field(
        default=None, description="Optional model routing hint for the LLM client"
    )

class PolicyRecommendationsResponse(BaseModel):
    """Response envelope for /policy/recommendations endpoint."""
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "job_id": "8b0a53e2-2b0f-4c2c-9f5c-8d8e3d9d2d0a",
                "result_path": "storage/8b0a53e2-2b0f-4c2c-9f5c-8d8e3d9d2d0a.policy.json",
                "emerging_count": 3,
                "has_recommendations": True,
            }
        },
    )

    job_id: str
    result_path: str
    emerging_count: int
    has_recommendations: bool


class UserResultItem(BaseModel):
    """User-scoped stored result item for frontend listing/detail views."""
    job_id: str
    status: Literal["done"]
    user_id: str
    result_path: str
    type: Literal["analysis", "policy"]
    source_job_id: Optional[str] = None
    message: Optional[str] = None
    content: Optional[Dict[str, Any]] = None

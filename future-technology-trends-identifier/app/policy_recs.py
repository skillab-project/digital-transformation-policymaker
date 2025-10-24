# -*- coding: utf-8 -*-
"""
Utilities to classify emerging technologies (vs. ESCO) and generate
policy recommendations per technology using an LLM.

This module:
1) Maps technologies to ESCO occupations/skills and flags "emerging" tech
2) Builds compact, evidence-rich prompts per tech
3) Calls an LLM concurrently to get JSON recommendations

Design goals:
- Deterministic, typed, and testable
- Safe dict access (no KeyError)
- Clear return schema (stable for API consumers)

Created on Thu Oct 23 13:33:15 2025

@author: tsoukj
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union, cast
from .config import settings
from .esco_match import map_technologies_to_esco_occupations, map_technologies_to_esco_skills, map_technologies_to_esco_both
from .llm_client import generate_json

# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------

class Target(str, Enum):
    OCCUPATIONS = "occupations"
    SKILLS = "skills"
    BOTH = "both"

class Match(TypedDict, total=False):
    """One ESCO match with a similarity score."""
    label: str
    score: Union[float, int]

class MappingEntry(TypedDict, total=False):
    """Mapping result for one technology to a side (skills or occupations)."""
    technology: str
    matches: List[Match]

class MappingBoth(TypedDict, total=False):
    """Combined mapping output when target='both'."""
    occupations: List[MappingEntry]
    skills: List[MappingEntry]

class Technology(TypedDict, total=False):
    """Minimal technology representation expected by this module."""
    name: str
    description: str
    domain: str

class ActionItem(TypedDict, total=False):
    area: str
    action: str
    rationale: str
    stakeholders: List[str]
    timeframe: str  # e.g., short/medium/long
    KPIs: List[str]
    risks: str
    priority: str

class RecommendationItem(TypedDict, total=False):
    technology: str
    actions: List[ActionItem]

class RecommendationsEnvelope(TypedDict, total=False):
    """LLM is asked to return {'recommendations': [...]}."""
    recommendations: List[RecommendationItem]

# ---------------------------------------------------------------------
# Helpers: emerging classification
# ---------------------------------------------------------------------
def _is_emerging_list(matches: Optional[List[Match]], threshold: float) -> bool:
    """
    Return True if a technology should be considered 'emerging'
    w.r.t. the provided matches list:
      - no matches -> emerging
      - all matches strictly below the similarity threshold -> emerging
    """
    if not matches:
        return True
    try:
        return not any(float(m.get("score", 0.0)) >= threshold for m in matches)
    except Exception:
        # Defensive: any malformed match list => consider emerging
        return True

def classify_emerging(technologies: List[Technology], target: Union[Target, str] = Target.BOTH, similarity_threshold: float = 0.5, top_n: int = 5) -> Tuple[List[Technology], MappingBoth]:
    """
    Map technologies to ESCO side(s) and return which should be treated as emerging.

    A technology is 'emerging' if it has:
      (a) no ESCO matches, or
      (b) all ESCO matches have similarity < similarity_threshold.

    Returns:
        emerging: list of tech dicts considered emerging
        mapping: combined mapping evidence in the form:
                 {'occupations': [...], 'skills': [...]}
    """
    tgt = Target(str(target))

    if tgt == Target.OCCUPATIONS:
        occ_maps: List[MappingEntry] = map_technologies_to_esco_occupations(
            technologies, top_n=top_n, threshold=0.0
        )
        emerging = [
            t
            for t, r in zip(technologies, occ_maps)
            if _is_emerging_list(cast(Optional[List[Match]], r.get("matches")), similarity_threshold)
        ]
        return emerging, {"occupations": occ_maps, "skills": []}

    if tgt == Target.SKILLS:
        skl_maps: List[MappingEntry] = map_technologies_to_esco_skills(
            technologies, top_n=top_n, threshold=0.0
        )
        emerging = [
            t
            for t, r in zip(technologies, skl_maps)
            if _is_emerging_list(cast(Optional[List[Match]], r.get("matches")), similarity_threshold)
        ]
        return emerging, {"occupations": [], "skills": skl_maps}

    # BOTH
    mapping: MappingBoth = map_technologies_to_esco_both(
        technologies, top_n=top_n, threshold=0.0
    )
    occ_maps = mapping.get("occupations", []) or []
    skl_maps = mapping.get("skills", []) or []

    emerging: List[Technology] = []
    for t, ro, rs in zip(technologies, occ_maps, skl_maps):
        occ_matches = cast(Optional[List[Match]], ro.get("matches")) if isinstance(ro, dict) else None
        skl_matches = cast(Optional[List[Match]], rs.get("matches")) if isinstance(rs, dict) else None
        if _is_emerging_list(occ_matches, similarity_threshold) or _is_emerging_list(
            skl_matches, similarity_threshold
        ):
            emerging.append(t)

    return emerging, mapping

# ---------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------
def _per_tech_query() -> str:
    """Static LLM instruction: schema + constraints."""
    return (
        'You are a policy analyst. Your task is to produce policy recommendations for the technology below. '
        'Return the output as a valid JSON object with the following structure:\n'
        '{\n'
        '  "recommendations": [\n'
        '    {\n'
        '      "technology": "Technology Name",\n'
        '      "actions": [\n'
        '        {\n'
        '          "area": "Training/Reskilling",\n'
        '          "action": "Concrete step",\n'
        '          "rationale": "Why this matters",\n'
        '          "stakeholders": ["Stakeholder 1", "Stakeholder 2"],\n'
        '          "timeframe": "short",\n'
        '          "KPIs": ["KPI 1"],\n'
        '          "risks": "Risk text",\n'
        '          "priority": "High"\n'
        '        }\n'
        '      ]\n'
        '    }\n'
        '  ]\n'
        '}\n'
        "Rules:\n"
        "- Output MUST be valid JSON. No prose/Markdown around it.\n"
        '- The top-level key MUST be "recommendations".\n'
        "- Include at least 1 recommendation and at least 2 actions.\n"
        "- Each action MUST include: area, action, rationale, timeframe, priority. Other fields are encouraged.\n"
        "- Use these areas where relevant: Training/Reskilling, Higher-Education Curricula, Funding Calls/Pilots, "
        "Standards/Interoperability, Incentives/Job Upgrading, Monitoring & KPIs.\n"
        "- Be specific and feasible in the EU labour market context."
    )

def _per_tech_context(similarity_threshold: float) -> str:
    """Extra context defining what 'emerging' means in this pipeline."""
    return (
        "An 'emerging technology' here means either (a) no ESCO match or (b) all ESCO matches "
        f"have similarity < {similarity_threshold}. Provide concise, actionable steps across "
        "Training/Reskilling, Higher-Education Curricula, Funding Calls/Pilots, "
        "Standards/Interoperability, Incentives/Job Upgrading, Monitoring & KPIs. "
        "Return ONLY valid JSON matching the given schema."
    )

def _per_tech_text(tech: Technology, occ_matches: List[Match], skl_matches: List[Match]) -> str:
    """Compact, evidence-rich content block for the LLM."""
    name = (tech.get("name") or "").strip()
    desc = (tech.get("description") or "").strip()
    dom = (tech.get("domain") or "").strip()

    def _fmt(matches: Optional[List[Match]]) -> str:
        if not matches:
            return "[]"
        top = matches[:5]
        safe = [
            {"label": str(m.get("label", "")), "score": round(float(m.get("score", 0.0)), 3)}
            for m in top
        ]
        return json.dumps(safe, ensure_ascii=False)

    return (
        "Technology:\n"
        f"- Name: {name}\n"
        f"- Domain: {dom}\n"
        f"- Description: {desc}\n\n"
        "ESCO evidence (for transparency; you may reference it but do not just paraphrase it):\n"
        f"- Occupation matches: {_fmt(occ_matches)}\n"
        f"- Skill matches: {_fmt(skl_matches)}\n"
    )

# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------
def generate_policy_recommendations(technologies: List[Technology], target: Union[Target, str] = Target.BOTH, similarity_threshold: float = 0.5, max_actions_per_tech: int = 4, llm_model: Optional[str] = None,   logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    For the subset of input technologies considered 'emerging',
    ask the LLM for policy recommendations and return a normalized, stable schema.

    Returns dict with keys:
        - emerging: List[Technology]
        - recommendations: List[RecommendationItem]
        - mapping_evidence: MappingBoth
    """
    log = logger or logging.getLogger(__name__)

    emerging, mapping_raw = classify_emerging(
        technologies,
        target=target,
        similarity_threshold=similarity_threshold,
        top_n=5,
    )

    occ_maps = mapping_raw.get("occupations", []) or []
    skl_maps = mapping_raw.get("skills", []) or []

    # Build per-tech jobs
    jobs: List[Tuple[Technology, str, str, str]] = []
    for idx, tech in enumerate(technologies):
        if tech not in emerging:
            continue

        # Safe indexed access: only when mapping lists are sufficiently long
        occ_matches: List[Match] = []
        if idx < len(occ_maps):
            occ_matches = cast(List[Match], occ_maps[idx].get("matches", [])) or []

        skl_matches: List[Match] = []
        if idx < len(skl_maps):
            skl_matches = cast(List[Match], skl_maps[idx].get("matches", [])) or []

        query = _per_tech_query()
        context = _per_tech_context(similarity_threshold)
        text = _per_tech_text(tech, occ_matches, skl_matches)
        jobs.append((tech, query, context, text))

    def _call(tech: Technology, query: str, context: str, text: str) -> List[RecommendationItem]:
        try:
            res = generate_json(query=query, context=context, text=text, timeout=settings.timeout)
            payload: RecommendationsEnvelope = res if isinstance(res, dict) else {"recommendations": []}
            items: List[RecommendationItem] = payload.get("recommendations", []) or []

            # Normalize each item
            for it in items:
                it.setdefault("technology", tech.get("name", ""))
                # Trim action list length (if present)
                actions = it.get("actions")
                if isinstance(actions, list) and len(actions) > max_actions_per_tech:
                    it["actions"] = actions[:max_actions_per_tech]
            return items

        except Exception as exc:
            log.warning("LLM call failed for tech '%s': %s", tech.get("name", ""), exc)
            return [{"technology": tech.get("name", ""), "actions": []}]

    # Concurrency: cap workers to avoid provider rate limits
    max_workers = max(1, min(settings.parallel_chunks, len(jobs)))
    merged: List[RecommendationItem] = []

    if jobs:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_call, *j): j[0] for j in jobs}
            for fut in as_completed(futures):
                merged.extend(fut.result())

    # Return a *flat list* of recommendations (no nested "recommendations" key)
    return {
        "emerging": emerging,
        "recommendations": merged,
        "mapping_evidence": mapping_raw,
    }

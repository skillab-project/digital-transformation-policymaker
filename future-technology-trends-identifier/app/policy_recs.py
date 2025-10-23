
from typing import List, Dict, Optional, Tuple
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from .esco_match import map_technologies_to_esco_occupations, map_technologies_to_esco_skills, map_technologies_to_esco_both
from .llm_client import generate_json
from .config import settings

# ----------------------------
# Emerging-tech identification
# ----------------------------
def _is_emerging_list(matches: Optional[List[Dict]], threshold: float) -> bool:
    if not matches:
        return True
    return not any(float(m.get("score", 0.0)) >= threshold for m in matches)

def classify_emerging(technologies: List[Dict], target: str = "both", similarity_threshold: float = 0.5, top_n: int = 5) -> Tuple[List[Dict], Dict[str, Dict]]:
    if target == "occupations":
        mapping = map_technologies_to_esco_occupations(technologies, top_n=top_n, threshold=0.0)
        emerging = [
            t for t, r in zip(technologies, mapping)
            if _is_emerging_list(r.get("matches"), similarity_threshold)
        ]
        return emerging, {"occupations": mapping, "skills": []}

    if target == "skills":
        mapping = map_technologies_to_esco_skills(technologies, top_n=top_n, threshold=0.0)
        emerging = [
            t for t, r in zip(technologies, mapping)
            if _is_emerging_list(r.get("matches"), similarity_threshold)
        ]
        return emerging, {"occupations": [], "skills": mapping}

    # both
    mapping = map_technologies_to_esco_both(technologies, top_n=top_n, threshold=0.0)
    occ_maps, skl_maps = mapping["occupations"], mapping["skills"]

    emerging = []
    for t, ro, rs in zip(technologies, occ_maps, skl_maps):
        occ_matches = ro.get("matches") if isinstance(ro, dict) else []
        skl_matches = rs.get("matches") if isinstance(rs, dict) else []
        if _is_emerging_list(occ_matches, similarity_threshold) or _is_emerging_list(skl_matches, similarity_threshold):
            emerging.append(t)

    return emerging, mapping

# ----------------------------
# Per-tech prompt builder
# ----------------------------
def _per_tech_query() -> str:
    return """
You are a policy analyst. Your task is to produce policy recommendations for the technology below.

Return the output as a valid JSON object with the following structure:

{
  "recommendations": [
    {
      "technology": "Technology Name",
      "actions": [
        {
          "area": "Training/Reskilling",
          "action": "Concrete step",
          "rationale": "Why this matters",
          "stakeholders": ["Stakeholder 1", "Stakeholder 2"],
          "timeframe": "short",
          "KPIs": ["KPI 1"],
          "risks": "Risk text",
          "priority": "High"
        }
      ]
    }
  ]
}

Rules:
- Output MUST be valid JSON. No prose/Markdown around it.
- The top-level key MUST be "recommendations".
- Include at least 1 recommendation and at least 2 actions.
- Each action MUST include: area, action, rationale, timeframe, priority. Other fields are encouraged.
- Use these areas where relevant: Training/Reskilling, Higher-Education Curricula, Funding Calls/Pilots,
  Standards/Interoperability, Incentives/Job Upgrading, Monitoring & KPIs.
- Be specific and feasible in the EU labour market context.
"""

def _per_tech_context(similarity_threshold: float) -> str:
    return (
        "An 'emerging technology' here means either (a) no ESCO match or (b) all ESCO matches "
        f"have similarity < {similarity_threshold}. Provide concise, actionable steps across "
        "Training/Reskilling, Higher-Education Curricula, Funding Calls/Pilots, "
        "Standards/Interoperability, Incentives/Job Upgrading, Monitoring & KPIs. "
        "Return ONLY valid JSON matching the given schema."
    )

def _per_tech_text(tech: Dict, occ_matches: List[Dict], skl_matches: List[Dict]) -> str:
    name = tech.get("name","").strip()
    desc = tech.get("description","").strip()
    dom  = tech.get("domain","").strip()

    def _fmt(matches: List[Dict]) -> str:
        if not matches: return "[]"
        # Compact evidence for the LLM
        top = matches[:5]
        return json.dumps([{"label": m["label"], "score": round(float(m["score"]), 3)} for m in top], ensure_ascii=False)

    return (
        f"Technology:\n"
        f"- Name: {name}\n"
        f"- Domain: {dom}\n"
        f"- Description: {desc}\n\n"
        f"ESCO evidence (for transparency; you may reference it but do not just paraphrase it):\n"
        f"- Occupation matches: {_fmt(occ_matches)}\n"
        f"- Skill matches: {_fmt(skl_matches)}\n"
    )

# ----------------------------
# Main entry
# ----------------------------
def generate_policy_recommendations(technologies: List[Dict], target: str = "both", similarity_threshold: float = 0.5, max_actions_per_tech: int = 4, llm_model: Optional[str] = None) -> Dict:
    emerging, mapping_raw = classify_emerging(
        technologies, target=target, similarity_threshold=similarity_threshold, top_n=5
    )

    occ_maps = mapping_raw.get("occupations", []) or []
    skl_maps = mapping_raw.get("skills", []) or []

    tasks = []
    for idx, tech in enumerate(technologies):
        if tech not in emerging:
            continue
        occ_matches = occ_maps[idx]["matches"] if idx < len(occ_maps) and occ_maps else []
        skl_matches = skl_maps[idx]["matches"] if idx < len(skl_maps) and skl_maps else []

        query   = _per_tech_query()
        context = _per_tech_context(similarity_threshold)
        text    = _per_tech_text(tech, occ_matches, skl_matches)
        tasks.append((tech, query, context, text))

    merged_recs: List[Dict] = []

    def _call(tech, query, context, text):
        try:
            res = generate_json(query=query, context=context, text=text, timeout=settings.timeout)
            # Normalize: ensure technology name is set
            items = res.get("recommendations", []) if isinstance(res, dict) else []
            for it in items:
                if not it.get("technology"):
                    it["technology"] = tech.get("name","")
                # Trim actions per tech
                if isinstance(it.get("actions"), list) and len(it["actions"]) > max_actions_per_tech:
                    it["actions"] = it["actions"][:max_actions_per_tech]
            return items
        except Exception:
            # Soft-fail: return empty recommendations for this tech
            return [{
                "technology": tech.get("name",""),
                "actions": []
            }]

    with ThreadPoolExecutor(max_workers=settings.parallel_chunks) as ex:
        fut_map = {ex.submit(_call, *t): t[0] for t in tasks}
        for fut in as_completed(fut_map):
            merged_recs.extend(fut.result())

    return {
        "emerging": emerging,
        "recommendations": {"recommendations": merged_recs},
        "mapping_evidence": mapping_raw
    }

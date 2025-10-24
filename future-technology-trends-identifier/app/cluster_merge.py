# -*- coding: utf-8 -*-
"""
Merge per-chunk technology extractions using semantic clustering.

Steps:
1) Encode "name + description" with a SentenceTransformer
2) Cluster embeddings via DBSCAN (cosine distance)
3) For each cluster:
   - Pick a representative technology (highest confidence, then longest description)
   - Union occupations, pick dominant domain
   - Keep related names and cluster size
4) Optionally filter out small clusters (e.g., singletons)

Returns:
    {"technologies": [ ...merged tech dicts... ]}

Created on Thu Oct 23 13:33:15 2025

@author: tsoukj
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Any, Dict, List, Tuple, TypedDict, Optional
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
try:
    # Optional: use project settings if available
    from .config import settings  # type: ignore
except Exception:  # pragma: no cover
    class _FallbackSettings:  # minimal defaults if settings not present
        embed_model = "all-MiniLM-L6-v2"
        embed_batch_size = 64
        dbscan_eps = 0.30
        dbscan_min_samples = 1
        cluster_min_size = 2
    settings = _FallbackSettings()  # type: ignore

# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------
class TechDict(TypedDict, total=False):
    name: str
    description: str
    domain: str
    occupations: List[str]
    confidence: float

# ---------------------------------------------------------------------
# Model loader (cached)
# ---------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_model(model_name: str) -> SentenceTransformer:
    """Load and cache the sentence transformer model once per process."""
    return SentenceTransformer(model_name)

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def merge_results(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Merge raw per-chunk extractions into clustered, de-duplicated technologies.

    Args:
        results: list of dicts, each possibly containing {"technologies": [ ... ]}

    Returns:
        {"technologies": [merged-tech, ...]}

    Notes:
        - Uses cosine distance DBSCAN with configurable eps/min_samples.
        - Clusters smaller than settings.cluster_min_size are dropped (default: 2).
        - Deterministic ordering: by (-cluster_size, name).
    """
    if not results:
        return {"technologies": []}
    
    # Flatten tech list
    all_techs: List[TechDict] = []
    for r in results:
        items = r.get("technologies") or []
        if isinstance(items, list):
            for t in items:
                if isinstance(t, dict) and t.get("name"):
                    all_techs.append(_sanitize_tech(t))

    if not all_techs:
        return {"technologies": []}

    # Texts to embed
    texts = [f"{t.get('name','')}. {t.get('description','')}".strip() for t in all_techs]

    # Encode
    model_name = getattr(settings, "embed_model", "all-MiniLM-L6-v2")
    batch_size = int(getattr(settings, "embed_batch_size", 64))
    model = _get_model(model_name)
    embeddings = model.encode(
        texts,
        convert_to_tensor=False,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=False,
    )

    # Cluster
    eps = float(getattr(settings, "dbscan_eps", 0.30))
    min_samples = int(getattr(settings, "dbscan_min_samples", 1))
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = db.fit_predict(embeddings)

    # Group by cluster id
    clusters: Dict[int, List[TechDict]] = {}
    for tech, lbl in zip(all_techs, labels):
        clusters.setdefault(int(lbl), []).append(tech)

    # Merge clusters
    merged: List[Dict[str, Any]] = []
    min_size = int(getattr(settings, "cluster_min_size", 2))

    for cid, members in clusters.items():
        if cid == -1:
            # DBSCAN noise points; treat as singletons
            if min_size > 1:
                continue
        if len(members) < min_size:
            continue

        rep = _pick_representative(members)
        occupations = _union_occupations(members)
        domain = _dominant_domain(members, fallback=rep.get("domain", ""))

        related = sorted(
            {m.get("name", "") for m in members if m.get("name", "") and m.get("name") != rep.get("name")}
        )

        confidence_values = [m.get("confidence") for m in members if isinstance(m.get("confidence"), (int, float))]
        confidence_max = max(confidence_values) if confidence_values else None
        # choose a single confidence field in output (max)
        rep_conf = confidence_max if confidence_max is not None else rep.get("confidence")

        merged.append(
            {
                "name": rep.get("name", ""),
                "description": rep.get("description", ""),
                "domain": domain,
                "occupations": occupations,
                "confidence": rep_conf,
                "cluster_size": len(members),
                "related_names": related,
            }
        )

    # Deterministic sort
    merged.sort(key=lambda x: (-int(x.get("cluster_size", 0)), str(x.get("name", "")).casefold()))
    return {"technologies": merged}

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _sanitize_tech(t: Dict[str, Any]) -> TechDict:
    """Ensure minimal fields and types; avoid KeyErrors downstream."""
    name = str(t.get("name", "")).strip()
    desc = str(t.get("description", "")).strip()
    domain = str(t.get("domain", "")).strip()

    occ = t.get("occupations")
    if isinstance(occ, list):
        occupations = sorted({str(o).strip() for o in occ if str(o).strip()})
    elif occ is None:
        occupations = []
    else:
        occupations = [str(occ).strip()] if str(occ).strip() else []

    conf = t.get("confidence")
    try:
        confidence = float(conf) if conf is not None else None
    except Exception:
        confidence = None

    out: TechDict = {
        "name": name,
        "description": desc,
        "domain": domain,
        "occupations": occupations,
    }
    if confidence is not None:
        out["confidence"] = confidence
    return out


def _pick_representative(members: List[TechDict]) -> TechDict:
    """
    Choose a representative tech:
        1) highest confidence
        2) longest description
        3) lexicographically smallest name (casefold)
    """
    def _score(t: TechDict) -> Tuple[float, int, str]:
        conf = t.get("confidence")
        # Use -len(desc) because we sort descending on first key;
        # but easier: return tuple with negative length? We'll sort with key and reverse=False
        desc_len = len(t.get("description", "") or "")
        name_key = (t.get("name") or "").casefold()
        # For sorting descending by conf/desc length, we return (-conf, -desc_len) if we sort ascending
        conf_sort = -(conf if isinstance(conf, (int, float)) else -1.0)
        return (conf_sort, -desc_len, name_key)

    # min by the tuple above (since we negated to simulate descending)
    return min(members, key=_score)


def _union_occupations(members: List[TechDict]) -> List[str]:
    """Union of occupations, normalized and sorted."""
    occs = set()
    for m in members:
        for o in m.get("occupations", []) or []:
            s = str(o).strip()
            if s:
                occs.add(s)
    return sorted(occs)


def _dominant_domain(members: List[TechDict], fallback: str = "") -> str:
    """Most common non-empty domain; fall back to provided value."""
    ctr = Counter([m.get("domain", "").strip() for m in members if m.get("domain", "").strip()])
    if not ctr:
        return fallback
    [(dom, _)] = ctr.most_common(1)
    return dom

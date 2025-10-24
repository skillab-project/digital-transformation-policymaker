# -*- coding: utf-8 -*-
"""
ESCO mapping utilities:
- Load ESCO Occupations/Skills CSVs
- Encode ESCO entries and query technologies
- Return top-N cosine-similarity matches

Public functions (stable shapes):
- map_technologies_to_esco_occupations(...)
- map_technologies_to_esco_skills(...)
- map_technologies_to_esco_both(...)
- warm_esco_caches()

Created on Thu Oct 23 13:33:15 2025

@author: tsoukj
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from .config import settings

# ---------------------------------------------------------------------
# Config & constants
# ---------------------------------------------------------------------
DEFAULT_MODEL_NAME = getattr(settings, "embed_model", "all-MiniLM-L6-v2")
BATCH_SIZE = int(getattr(settings, "embed_batch_size", 128))
CACHE_DIR = Path(getattr(settings, "esco_cache_dir", "storage/esco_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------
_MODEL: SentenceTransformer | None = None

def _get_model() -> SentenceTransformer:
    """Load the SentenceTransformer once per process (lazy)."""
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(DEFAULT_MODEL_NAME)
    return _MODEL

def _embed(texts: List[str], normalize: bool = True) -> np.ndarray:
    """
    Encode a list of texts into embeddings (float32).
    If `normalize=True`, cosine similarity reduces to dot product.
    """
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)  # 384 fits MiniLM; harmless for empties
    model = _get_model()
    vecs = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
        batch_size=BATCH_SIZE,
    )
    return np.asarray(vecs, dtype=np.float32)

# ---------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------
def _load_df(csv_path: str) -> pd.DataFrame:
    """
    Load ESCO CSV and ensure required columns exist.

    Expected columns:
      - label (str)
      - alternative_labels (str, optional; comma/semicolon-separated)
      - description (str, optional)
      - uri/id (optional, if present we propagate it in results)
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"ESCO CSV not found: {csv_path}")

    df = pd.read_csv(p)
    for col in ("label", "alternative_labels", "description"):
        if col not in df.columns:
            df[col] = ""
    # Normalize types and NaNs
    df = df.fillna({"label": "", "alternative_labels": "", "description": ""})
    df["label"] = df["label"].astype(str)
    df["alternative_labels"] = df["alternative_labels"].astype(str)
    df["description"] = df["description"].astype(str)
    return df

def _to_texts(df: pd.DataFrame) -> List[str]:
    """
    Build the text to embed per ESCO row:
        "label. alternative_labels. description"
    """
    return (df["label"] + ". " + df["alternative_labels"] + ". " + df["description"]).tolist()

# ---------------------------------------------------------------------
# Disk persistence
# ---------------------------------------------------------------------
def _cache_key(csv_path: str) -> str:
    p = Path(csv_path).resolve()
    meta = f"{DEFAULT_MODEL_NAME}|{str(p)}|{p.stat().st_mtime}"
    import hashlib

    return hashlib.sha256(meta.encode("utf-8")).hexdigest()

def _load_or_build_embeddings(csv_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load (or compute & persist) embeddings for a given ESCO CSV.
    Returns a (df, embeddings) pair. Embeddings are memory-mapped.
    """
    key = _cache_key(csv_path)
    df_file = CACHE_DIR / f"{key}.df.parquet"
    emb_file = CACHE_DIR / f"{key}.embeddings.npy"

    if df_file.exists() and emb_file.exists():
        df = pd.read_parquet(df_file)
        embs = np.load(emb_file, mmap_mode="r")
        return df, embs

    df = _load_df(csv_path)
    texts = _to_texts(df)
    embs = _embed(texts, normalize=True)

    df.to_parquet(df_file, index=False)
    np.save(emb_file, embs)
    # Reload as memmap to reduce RSS and allow slicing without full load
    embs = np.load(emb_file, mmap_mode="r")
    return df, embs

# ---------------------------------------------------------------------
# In-memory caches
# ---------------------------------------------------------------------
_cached: Dict[str, Any] = {"occ_df": None, "occ_embs": None, "sk_df": None, "sk_embs": None}

def _ensure_occ_cache():
    if _cached["occ_df"] is None or _cached["occ_embs"] is None:
        df, embs = _load_or_build_embeddings(settings.esco_occupations_csv)
        _cached["occ_df"], _cached["occ_embs"] = df, embs
    return _cached["occ_df"], _cached["occ_embs"]

def _ensure_skills_cache():
    if _cached["sk_df"] is None or _cached["sk_embs"] is None:
        df, embs = _load_or_build_embeddings(settings.esco_skills_csv)
        _cached["sk_df"], _cached["sk_embs"] = df, embs
    return _cached["sk_df"], _cached["sk_embs"]

# ---------------------------------------------------------------------
# Core top-N search
# ---------------------------------------------------------------------
def _topn_search(query_vec: np.ndarray, index_vecs: np.ndarray, df: pd.DataFrame, top_n: int, threshold: float) -> List[Dict[str, Any]]:
    """
    Find top-N cosine-similarity matches for a SINGLE query vector
    against a normalized embedding matrix (index_vecs).
    """
    if index_vecs.size == 0:
        return []

    # Cosine == dot because both sides are normalized
    sims = index_vecs @ query_vec  # (N,)
    n = sims.shape[0]
    k = min(max(top_n, 0), n)

    if k == 0:
        return []

    # Argpartition is O(N); then sort those k by similarity desc
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]

    matches: List[Dict[str, Any]] = []
    for i in idx:
        score = float(sims[i])
        if score < threshold:
            # because we sorted by desc sim, we can early break here
            continue
        row = df.iloc[int(i)]
        item: Dict[str, Any] = {"label": row.get("label", ""), "score": round(score, 6)}
        # propagate optional identifiers if available
        for opt in ("uri", "id", "esco_uri", "esco_id"):
            if opt in row:
                item[opt] = row[opt]
        matches.append(item)

    return matches

# ---------------------------------------------------------------------
# Public mapping functions
# ---------------------------------------------------------------------
def map_technologies_to_esco_occupations(technologies: List[Dict[str, Any]], top_n: int = 5, threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Map each technology to ESCO Occupations.

    Returns a list of:
      {"technology": <name>, "matches": [{"label": str, "score": float, ...}, ...]}
    """
    if not technologies:
        return []

    occ_df, occ_embs = _ensure_occ_cache()

    tech_texts = [
        (
            f"{t.get('name','')}. {t.get('description','')}. {t.get('domain','')}. "
            f"{' ; '.join(t.get('occupations', []) or [])}"
        )
        for t in technologies
    ]
    tech_vecs = _embed(tech_texts, normalize=True)  # (T, d)

    results = []
    for tech, vec in zip(technologies, tech_vecs):
        matches = _topn_search(vec, occ_embs, occ_df, top_n, threshold)
        results.append({"technology": tech.get("name", ""), "matches": matches})
    return results

def map_technologies_to_esco_skills(technologies: List[Dict[str, Any]], top_n: int = 5, threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Map each technology to ESCO Skills.

    Returns a list of:
      {"technology": <name>, "matches": [{"label": str, "score": float, ...}, ...]}
    """
    if not technologies:
        return []

    sk_df, sk_embs = _ensure_skills_cache()

    tech_texts = [
        (
            f"{t.get('name','')}. {t.get('description','')}. {t.get('domain','')}. "
            f"{' ; '.join(t.get('occupations', []) or [])}"
        )
        for t in technologies
    ]
    tech_vecs = _embed(tech_texts, normalize=True)  # (T, d)

    results = []
    for tech, vec in zip(technologies, tech_vecs):
        matches = _topn_search(vec, sk_embs, sk_df, top_n, threshold)
        results.append({"technology": tech.get("name", ""), "matches": matches})
    return results

def map_technologies_to_esco_both(technologies: List[Dict[str, Any]], top_n: int = 5, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Map to both ESCO sides:
        {"occupations": [...], "skills": [...]}
    """
    return {
        "occupations": map_technologies_to_esco_occupations(technologies, top_n, threshold),
        "skills": map_technologies_to_esco_skills(technologies, top_n, threshold),
    }

# ---------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------
def warm_esco_caches() -> None:
    """Build/load both ESCO embeddings at startup so first request is instant."""
    _ensure_occ_cache()
    _ensure_skills_cache()

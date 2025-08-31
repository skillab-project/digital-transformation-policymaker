from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from .config import settings

# -------- Model (single load) --------
MODEL = SentenceTransformer("all-mpnet-base-v2")

def _embed(texts: List[str], normalize: bool = True):
    vecs = MODEL.encode(texts, convert_to_tensor=False, normalize_embeddings=normalize)
    return np.asarray(vecs, dtype=np.float32)

# -------- CSV loaders --------
def _load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # columns expected: label, alternative_labels, description
    for col in ("label", "alternative_labels", "description"):
        if col not in df.columns:
            df[col] = ""
    df["alternative_labels"] = df["alternative_labels"].fillna("")
    df["description"] = df["description"].fillna("")
    return df

def _to_texts(df: pd.DataFrame) -> List[str]:
    return [
        f"{row['label']}. {row.get('alternative_labels','')}. {row.get('description','')}"
        for _, row in df.iterrows()
    ]

# -------- In-memory caches --------
_cached = {
    "occ_df": None, "occ_embs": None,
    "sk_df": None,  "sk_embs": None
}

def _ensure_occ_cache():
    if _cached["occ_df"] is None or _cached["occ_embs"] is None:
        df = _load_df(settings.esco_occupations_csv)
        embs = _embed(_to_texts(df), normalize=True)
        _cached["occ_df"], _cached["occ_embs"] = df, embs
    return _cached["occ_df"], _cached["occ_embs"]

def _ensure_skills_cache():
    if _cached["sk_df"] is None or _cached["sk_embs"] is None:
        df = _load_df(settings.esco_skills_csv)
        embs = _embed(_to_texts(df), normalize=True)
        _cached["sk_df"], _cached["sk_embs"] = df, embs
    return _cached["sk_df"], _cached["sk_embs"]

# -------- Core top-N search --------
def _topn_search(query_vec: np.ndarray, index_vecs: np.ndarray, df: pd.DataFrame,
                 top_n: int, threshold: float):
    # cosine == dot when normalized
    sims = index_vecs @ query_vec  # (N,)
    idx = np.argpartition(-sims, min(top_n, len(sims)-1))[:top_n]
    idx = idx[np.argsort(-sims[idx])]

    matches = []
    for i in idx:
        score = float(sims[i])
        if score < threshold:
            continue
        row = df.iloc[int(i)]
        matches.append({"label": row["label"], "score": score})
    return matches

# -------- Public functions --------
def map_technologies_to_esco_occupations(technologies: List[Dict], top_n: int = 5, threshold: float = 0.5):
    occ_df, occ_embs = _ensure_occ_cache()
    tech_texts = [
        f"{t.get('name','')}. {t.get('description','')}. {t.get('domain','')}. {'; '.join(t.get('occupations', []))}"
        for t in technologies
    ]
    tech_vecs = _embed(tech_texts, normalize=True)  # (T, d)

    results = []
    for tech, vec in zip(technologies, tech_vecs):
        matches = _topn_search(vec, occ_embs, occ_df, top_n, threshold)
        results.append({"technology": tech.get("name",""), "matches": matches})
    return results

def map_technologies_to_esco_skills(technologies: List[Dict], top_n: int = 5, threshold: float = 0.5):
    sk_df, sk_embs = _ensure_skills_cache()
    tech_texts = [
        f"{t.get('name','')}. {t.get('description','')}. {t.get('domain','')}. {'; '.join(t.get('occupations', []))}"
        for t in technologies
    ]
    tech_vecs = _embed(tech_texts, normalize=True)  # (T, d)

    results = []
    for tech, vec in zip(technologies, tech_vecs):
        matches = _topn_search(vec, sk_embs, sk_df, top_n, threshold)
        results.append({"technology": tech.get("name",""), "matches": matches})
    return results

def map_technologies_to_esco_both(technologies: List[Dict], top_n: int = 5, threshold: float = 0.5):
    return {
        "occupations": map_technologies_to_esco_occupations(technologies, top_n, threshold),
        "skills":      map_technologies_to_esco_skills(technologies, top_n, threshold),
    }

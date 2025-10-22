
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import hashlib
from pathlib import Path
from sentence_transformers import SentenceTransformer
from .config import settings

# -------- Configuration --------
MODEL_NAME = "all-MiniLM-L6-v2"
MODEL = SentenceTransformer(MODEL_NAME)
CACHE_DIR = Path("storage/esco_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _embed(texts: List[str], normalize: bool = True):
    vecs = MODEL.encode(texts, convert_to_numpy=True, normalize_embeddings=normalize, show_progress_bar=False, batch_size=128)
    return np.asarray(vecs, dtype=np.float32)

# -------- CSV loaders --------
def _load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # columns expected: label, alternative_labels, description
    for col in ("label", "alternative_labels", "description"):
        if col not in df.columns:
            df[col] = ""
    return df.fillna({"alternative_labels": "", "description": ""})

def _to_texts(df: pd.DataFrame) -> List[str]:
    return (df["label"].fillna("") + ". " + df["alternative_labels"].fillna("") + ". " + df["description"].fillna("")).tolist()

# -------- Disk persistence --------
def _cache_key(csv_path: str) -> str:
    p = Path(csv_path).resolve()
    meta = f"{MODEL_NAME}|{p}|{p.stat().st_mtime}"
    return hashlib.sha256(meta.encode("utf-8")).hexdigest()

def _load_or_build_embeddings(csv_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
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
    embs = np.load(emb_file, mmap_mode="r")  # reload as memory-mapped
    return df, embs

# -------- In-memory caches --------
_cached = {"occ_df": None, "occ_embs": None, "sk_df": None,  "sk_embs": None}

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

# -------- Warm-up on startup --------
def warm_esco_caches():
    """
    Build/load both ESCO embeddings at startup so first request is instant.
    """
    _ensure_occ_cache()
    _ensure_skills_cache()

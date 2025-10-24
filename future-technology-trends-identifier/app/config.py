# -*- coding: utf-8 -*-
"""
Runtime settings for the Future Tech Trends Analyzer.

Loads from environment (.env) via pydantic-settings and exposes a single
Settings instance: `settings`.

Key features:
- Input validation (e.g., overlap < chunk_size)
- Path normalization and directory creation (storage, ESCO cache)
- Extra knobs for embedding model, clustering, progress bars, etc.

Created on Thu Oct 23 13:33:15 2025

@author: tsoukj
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        protected_namespaces=("settings_",),
        case_sensitive=False,
    )

    # ================ API Configuration ================
    api_token: str = Field("", env="API_TOKEN", description="Bearer token for LLM API")
    api_url: str = Field("http://localhost:3000", env="API_URL")
    model_name: str = Field("mistral:latest", env="MODEL_NAME")  # chat/completions model
    temperature: float = Field(0.0, ge=0.0, le=2.0, env="TEMPERATURE")
    seed: int = Field(42, ge=0, env="SEED")
    timeout: int = Field(120, ge=1, env="TIMEOUT")  # seconds per LLM call
    extra_headers: Optional[Dict[str, str]] = Field(default=None, env="EXTRA_HEADERS_JSON")

    # ================ Chunking / parsing ================
    chunk_size: int = Field(6000, ge=256, env="CHUNK_SIZE")
    overlap: int = Field(400, ge=0, env="OVERLAP")
    max_pages: int = Field(0, ge=0, env="MAX_PAGES")  # 0 = all pages
    skip_pages: str = Field("-1", env="SKIP_PAGES")  # e.g., "1,2,-1"

    # Large-document heuristics
    priority_keywords: str = Field("Horizon Europe,RIA,IA,work program,WP", env="PRIORITY_KEYWORDS")
    exclude_sections: str = Field("references,appendix,bibliography", env="EXCLUDE_SECTIONS")

    # ================ Performance ======================
    parallel_chunks: int = Field(1, ge=1, le=64, env="PARALLEL_CHUNKS")
    progress_bar: bool = Field(False, env="PROGRESS_BAR")  # only used in CLI/dev

    # ================ Embeddings / ESCO ================
    embed_model: str = Field("all-MiniLM-L6-v2", env="EMBED_MODEL")
    embed_batch_size: int = Field(128, ge=1, le=1024, env="EMBED_BATCH_SIZE")
    esco_cache_dir: str = Field("storage/esco_cache", env="ESCO_CACHE_DIR")
    esco_occupations_csv: str = Field("datasets/all_occupations.csv", env="ESCO_OCCUPATIONS_CSV")
    esco_skills_csv: str = Field("datasets/all_skills.csv", env="ESCO_SKILLS_CSV")

    # DBSCAN clustering (cluster_merge.py)
    dbscan_eps: float = Field(0.30, ge=0.01, le=1.0, env="DBSCAN_EPS")
    dbscan_min_samples: int = Field(1, ge=1, le=50, env="DBSCAN_MIN_SAMPLES")
    cluster_min_size: int = Field(2, ge=1, le=100, env="CLUSTER_MIN_SIZE")

    # ================ Files / Storage ==================
    datasets_folder: str = Field("datasets", env="DATASETS_FOLDER")
    storage_dir: str = Field("storage", env="STORAGE_DIR")

    # ---------------- Validators & Post-processing ----------------

    @model_validator(mode="after")
    def _validate_and_normalize(self):
        # Ensure overlap < chunk_size (reduce overlap if misconfigured)
        if self.overlap >= self.chunk_size:
            # Keep it safe, reduce overlap to 1/4 of chunk_size
            object.__setattr__(self, "overlap", max(0, self.chunk_size // 4))

        # Normalize directories to absolute paths and ensure they exist
        storage = Path(self.storage_dir).resolve()
        esco_cache = Path(self.esco_cache_dir).resolve()
        datasets = Path(self.datasets_folder).resolve()

        storage.mkdir(parents=True, exist_ok=True)
        esco_cache.mkdir(parents=True, exist_ok=True)
        datasets.mkdir(parents=True, exist_ok=True)

        object.__setattr__(self, "storage_dir", str(storage))
        object.__setattr__(self, "esco_cache_dir", str(esco_cache))
        object.__setattr__(self, "datasets_folder", str(datasets))

        return self

    @field_validator("api_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @field_validator("priority_keywords", "exclude_sections", mode="before")
    @classmethod
    def _coerce_csv_str(cls, v):
        # accept lists passed via env (e.g., JSON) but keep as comma string for current callers
        if isinstance(v, (list, tuple)):
            return ",".join(str(x) for x in v)
        return v

settings = Settings()

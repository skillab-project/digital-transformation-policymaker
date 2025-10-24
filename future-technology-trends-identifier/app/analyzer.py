# -*- coding: utf-8 -*-
"""
PDF analysis pipeline for Future Tech Trends.

Steps:
1) Extract + clean text from PDF
2) Detect sections and chunk (or fallback to naive chunking)
3) Analyze chunks concurrently via LLM
4) Merge per-chunk extractions into a single result

Return value is the merged dict that `merge_results()` produces
(typically: {"technologies": [...]}).

Created on Thu Oct 23 13:33:15 2025

@author: tsoukj
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TypedDict
from .config import settings
from .cluster_merge import merge_results
from .llm_client import analyze_chunk
from .pdf_processor import chunk_text, clean_text, detect_sections, extract_text
try:
    from tqdm import tqdm  # optional, only used when enabled
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------
class Chunk(TypedDict, total=False):
    text: str
    context: str
    section: str
    byte_size: int

class ChunkAnalysis(TypedDict, total=False):
    # Shape produced by analyze_chunk() per chunk (flexible—pass-through to merge)
    technologies: List[Dict[str, Any]]
    section: str

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
DEFAULT_QUERY = (
    "You are an expert technology analyst. Your task is to read the following document chunk and "
    "identify explicitly mentioned, job-relevant, general-purpose emerging technologies.\n\n"
    "The technologies you extract must be:\n"
    " - Explicitly named or clearly described in the document\n"
    " - Broad in scope\n"
    " - Relevant to future workforce needs and skills demand (i.e., tied to roles, processes, or applications in industry)\n\n"
    "For each valid technology explicitly mentioned, extract and provide the following information:\n"
    "1. Name: Clearly name the technology (be specific, e.g., “Post-quantum cryptography” not just “AI”)\n"
    "2. Description: A concise description (50–100 words) explaining what the technology is and how it is used or proposed in the document.\n"
    "3. Domain: Identify the domain or sector where the technology applies (e.g., ICT, Health, Energy, Manufacturing).\n"
    "4. Future Occupations Needed: List the specific roles or job titles that will be needed to support or develop this technology in the future.\n"
    "5. Confidence Score (1–5): Based on the context of the document, rate how explicitly the technology was mentioned (5 = clearly described and emphasized, 1 = vague or only implied).\n\n"
    "Return the output as a valid JSON object with the following structure, always including all five fields:\n"
    '{\n  "technologies": [\n    {\n'
    '      "name": "Technology Name",\n'
    '      "description": "Description here",\n'
    '      "domain": "Domain here",\n'
    '      "occupations": ["Occupation 1", "Occupation 2"],\n'
    '      "confidence": 5\n'
    "    }\n  ]\n}\n\n"
    "❌ Do not invent technologies based on your own knowledge.\n"
    "❌ Do not include anything that is inferred or only implied."
)

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def process_pdf(pdf_path: str, query: Optional[str] = None, max_chunks: int = 100) -> Dict[str, Any]:
    """
    Process a PDF and return merged per-chunk extractions.

    Args:
        pdf_path: Path to a PDF file.
        query: Optional custom LLM instruction; falls back to DEFAULT_QUERY.
        max_chunks: Hard cap on analyzed chunks (after chunking). Use None/0 to analyze all.

    Returns:
        Dict with merged results (the exact schema is defined by merge_results()).

    Notes:
        - Concurrency is capped by settings.parallel_chunks and the number of chunks.
        - If section detection fails, falls back to naive fixed-size chunking with overlap.
        - Any per-chunk analysis failure is logged and skipped (soft-fail).
    """
    # 1) Extract and clean
    raw_text, page_map = extract_text(pdf_path)  # page_map currently unused, kept for future use
    cleaned = clean_text(raw_text)

    # 2) Section-aware chunking (fallback to naive)
    sections = detect_sections(cleaned)
    if sections:
        chunks: List[Chunk] = chunk_text(cleaned, sections)
    else:
        chunks = _naive_chunks(cleaned)

    # Respect max_chunks if provided
    use_chunks = chunks[:max_chunks] if max_chunks else chunks
    if not use_chunks:
        log.warning("No chunks produced for pdf=%s", pdf_path)
        return {"technologies": []}  # empty, valid shape for downstream

    # 3) Analyze each chunk concurrently
    q = query or DEFAULT_QUERY
    max_workers = max(1, min(settings.parallel_chunks, len(use_chunks)))
    analyses: List[ChunkAnalysis] = []

    # We keep original order by storing the index and sorting results by it
    def _submit_jobs(executor: ThreadPoolExecutor) -> Dict[Future, int]:
        jobs: Dict[Future, int] = {}
        for idx, ch in enumerate(use_chunks):
            fut = executor.submit(
                analyze_chunk,
                ch.get("text", ""),
                q,
                ch.get("context", "Full document"),
                settings.timeout,
            )
            jobs[fut] = idx
        return jobs

    iterator: Iterable = as_completed  # for typing
    total = len(use_chunks)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = _submit_jobs(ex)

        # Optional tqdm progress (only when available AND enabled)
        if getattr(settings, "progress_bar", False) and tqdm is not None:
            iterator = lambda fs: tqdm(as_completed(fs), total=total, desc="Analyzing chunks")  # type: ignore[misc]

        for fut in iterator(futures):
            idx = futures[fut]
            ch = use_chunks[idx]
            try:
                out = fut.result()
                if not out:
                    continue
                # Annotate with section for downstream merging transparency
                if isinstance(out, dict):
                    out["section"] = ch.get("section", "No section")
                    analyses.append(out)  # type: ignore[arg-type]
                else:
                    log.warning("Unexpected analyze_chunk() output type at idx=%s: %r", idx, type(out))
            except Exception as exc:  # noqa: BLE001
                log.warning("Chunk analysis failed at idx=%s: %s", idx, exc)

    # 4) Merge and return
    merged = merge_results(analyses)
    return merged

def save_json(data: Dict[str, Any], out_path: str) -> None:
    """Persist dict to pretty JSON (UTF-8)."""
    Path(out_path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON file into a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------
def _naive_chunks(text: str) -> List[Chunk]:
    """
    Fallback fixed-size chunking with overlap.

    Uses settings.chunk_size and settings.overlap and guards against
    invalid configurations (e.g., overlap >= chunk_size).
    """
    chunk_size = max(1, int(getattr(settings, "chunk_size", 3000)))
    overlap = int(getattr(settings, "overlap", 300))

    if overlap >= chunk_size:
        log.warning("overlap (%s) >= chunk_size (%s); reducing overlap to chunk_size//4", overlap, chunk_size)
        overlap = max(0, chunk_size // 4)

    chunks: List[Chunk] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(n, start + chunk_size)
        chunk_text_str = text[start:end]
        chunks.append(
            {
                "text": chunk_text_str,
                "context": "Full document",
                "section": "No section",
                "byte_size": len(chunk_text_str.encode("utf-8")),
            }
        )
        if end >= n:
            break
        # Advance with overlap
        start = end - overlap
        if start < 0:  # extra safety
            start = 0

    return chunks

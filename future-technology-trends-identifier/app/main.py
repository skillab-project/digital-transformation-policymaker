# -*- coding: utf-8 -*-
"""
FastAPI entrypoint for the Future Tech Trends Analyzer.

Responsibilities:
- Accept PDF uploads and spawn background analysis jobs
- Expose job status and result download endpoints
- Map extracted technologies to ESCO (occupations/skills/both)
- Generate policy recommendations for *emerging* technologies

Design notes:
- Duplicate PDF detection via SHA-256 over file bytes
- Background task for CPU/IO heavy PDF processing
- Storage under ./storage (configurable if you later add settings)
- Response models use your existing Pydantic types (JobStatus, ESCOMapRequest, PolicyReq)

Created on Thu Oct 23 13:33:15 2025

@author: tsoukj
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import uuid
from typing import Any, Optional, Union
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from .analyzer import load_json, process_pdf, save_json
from .esco_match import map_technologies_to_esco_occupations, map_technologies_to_esco_skills, map_technologies_to_esco_both, warm_esco_caches
from .jobs import new_job, set_status, get_job, list_jobs, rehydrate_from_storage, _load_jobs
from .models import ESCOMapRequest, ESCOMapItem, ESCOMapBoth, JobStatus, PolicyReq, PolicyRecommendationsResponse, PolicyGeneratorOutput
from .policy_recs import generate_policy_recommendations

# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Future Tech Trends Analyzer",
    version="1.0.0",
    openapi_tags=[
        {"name": "health", "description": "Service health checks"},
        {"name": "analysis", "description": "PDF analysis and job management"},
        {"name": "mapping", "description": "ESCO mapping utilities"},
        {"name": "policy", "description": "Policy recommendations"},
    ],
)

log = logging.getLogger(__name__)

STORAGE = Path("storage")
STORAGE.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _sha256(data: bytes) -> str:
    """Return the hex SHA-256 digest of the given bytes."""
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

# -----------------------------------------------------------------------------
# Lifecycle
# -----------------------------------------------------------------------------
@app.on_event("startup")
def _startup() -> None:
    """
    Initialize in-memory job cache and ESCO caches on service startup.

    Notes:
        - `_load_jobs()` may restore in-memory tracking structures.
        - `rehydrate_from_storage()` associates any previously saved results.
        - `warm_esco_caches()` primes ESCO lookup layers for faster mapping.
    """
    _load_jobs()
    rehydrate_from_storage(str(STORAGE))
    warm_esco_caches()
    log.info("Startup complete: storage=%s", STORAGE.resolve())

# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------
@app.get("/health", tags=["health"])
def health() -> dict[str, bool]:
    """Simple health ping."""
    return {"ok": True}

# -----------------------------------------------------------------------------
# Analysis: upload & jobs
# -----------------------------------------------------------------------------
@app.post("/analyze/pdf", response_model=JobStatus, tags=["analysis"])
async def analyze_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...), query: Optional[str] = None) -> JobStatus:
    """
    Enqueue PDF analysis.

    Behavior:
        - Validates .pdf extension and non-empty content
        - Deduplicates by content hash: if a finished job with the same hash exists,
          returns its job status immediately
        - Otherwise, stores the PDF and spawns a background task that:
            * processes the PDF
            * writes result JSON
            * updates job status accordingly
    """
    filename = (file.filename or "").lower()
    if not filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")

    file_hash = _sha256(data)

    # Deduplicate by hash if a completed job exists
    for jid, info in (list_jobs() or {}).items():
        if (
            info.get("status") == "done"
            and info.get("file_hash") == file_hash
            and info.get("result_path")
        ):
            return JobStatus(
                job_id=jid,
                status="done",
                message="Duplicate PDF detected. Reusing previous result.",
                result_path=info["result_path"],
            )

    # New job
    job_id = new_job()
    tmp_path = STORAGE / f"{job_id}.pdf"
    tmp_path.write_bytes(data)

    def _run() -> None:
        """Background task: process the PDF and persist the analysis JSON."""
        try:
            set_status(job_id, "running", message="Processing PDF", file_hash=file_hash)
            result: dict[str, Any] = process_pdf(str(tmp_path), query=query)

            out_path = STORAGE / f"{job_id}.analysis.json"
            save_json(result, str(out_path))

            set_status(job_id, "done", result_path=str(out_path), file_hash=file_hash)
        except Exception as exc:  # noqa: BLE001 - bubble through as status
            log.exception("PDF analysis failed for job %s: %s", job_id, exc)
            set_status(job_id, "error", message=str(exc), file_hash=file_hash)

    background_tasks.add_task(_run)
    return JobStatus(job_id=job_id, status="queued")

@app.get("/jobs/{job_id}", response_model=JobStatus, tags=["analysis"])
def job_status(job_id: str) -> JobStatus:
    """Return the status of a job by its ID."""
    info = get_job(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(job_id=job_id, **info)

@app.get("/results/{job_id}/download", tags=["analysis"])
def download_result(job_id: str) -> FileResponse:
    """
    Download the JSON result of a completed job.

    Returns:
        application/json file with a stable filename: {job_id}.analysis.json
    """
    info = get_job(job_id)
    if not info or info.get("status") != "done":
        raise HTTPException(status_code=404, detail="Job not found or not done")

    path = info.get("result_path")
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    return FileResponse(
        path,
        media_type="application/json",
        filename=f"{job_id}.analysis.json",
    )

# -----------------------------------------------------------------------------
# ESCO mapping
# -----------------------------------------------------------------------------
@app.post("/map-to-esco", tags=["mapping"], response_model=Union[list[ESCOMapItem], ESCOMapBoth])
def map_to_esco(req: ESCOMapRequest) -> Union[list[ESCOMapItem], ESCOMapBoth]:
    """
    Map provided (or job-derived) technologies to ESCO.

    Request:
        - Either `technologies` (inline) OR `job_id` (to fetch technologies from analysis)
        - `target` in {"occupations", "skills", "both"}
        - `top_n`, `threshold` forwarded to the mapping layer

    Returns:
        Mapping list(s) or a dict with both sides, depending on target.
    """
    if not req.technologies and not req.job_id:
        raise HTTPException(status_code=400, detail="Provide either job_id or technologies.")

    techs = req.technologies
    if req.job_id:
        info = get_job(req.job_id)
        if not info or info.get("status") != "done":
            raise HTTPException(status_code=404, detail="Job not found or not done")

        data = load_json(info.get("result_path"))
        techs = data.get("technologies", [])

    if req.target == "occupations":
        return map_technologies_to_esco_occupations(techs, top_n=req.top_n, threshold=req.threshold)
    if req.target == "skills":
        return map_technologies_to_esco_skills(techs, top_n=req.top_n, threshold=req.threshold)

    # both
    return map_technologies_to_esco_both(techs, top_n=req.top_n, threshold=req.threshold)

# -----------------------------------------------------------------------------
# Policy recommendations
# -----------------------------------------------------------------------------
@app.post("/policy/recommendations", tags=["policy"], response_model=PolicyRecommendationsResponse)
def policy_recommendations(req: PolicyReq) -> PolicyRecommendationsResponse:
    """
    Generate policy recommendations for *emerging* technologies.

    Behavior:
        - If `technologies` is empty and `job_id` is provided, source tech list from job result
        - Calls `generate_policy_recommendations` which:
            * classifies emerging vs ESCO evidence
            * calls LLM per emerging tech
            * returns a flat list of recommendations + evidence
        - Persists output as {job_id}.policy.json (uses provided job_id or a new UUID)

    Returns:
        Envelope with:
            - job_id
            - result_path
            - emerging_count
            - has_recommendations (bool)
    """
    techs = req.technologies or []
    if not techs and req.job_id:
        info = get_job(req.job_id)
        if not info or info.get("status") != "done":
            raise HTTPException(status_code=404, detail="Job not found or not done")
        data = load_json(info.get("result_path"))
        techs = data.get("technologies", [])

    if not techs:
        raise HTTPException(status_code=400, detail="No technologies provided.")

    out = generate_policy_recommendations(
        technologies=techs,
        target=req.target,
        similarity_threshold=req.similarity_threshold,
        max_actions_per_tech=req.max_actions_per_tech,
        llm_model=req.llm_model,
    )

    out_job_id = req.job_id or str(uuid.uuid4())
    out_path = STORAGE / f"{out_job_id}.policy.json"
    save_json(out, str(out_path))

    has_recs = bool(out.get("recommendations") or [])

    return {
        "job_id": out_job_id,
        "result_path": str(out_path),
        "emerging_count": len(out.get("emerging", [])),
        "has_recommendations": has_recs,
    }

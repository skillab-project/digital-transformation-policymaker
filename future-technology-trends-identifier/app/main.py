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
import threading
from datetime import datetime, timezone
from pathlib import Path
import uuid
from typing import Any, Optional, Union
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from .analyzer import load_json, process_pdf, save_json
from .esco_match import map_technologies_to_esco_occupations, map_technologies_to_esco_skills, map_technologies_to_esco_both, warm_esco_caches
from .jobs import new_job, set_status, get_job, list_jobs, delete_job, rehydrate_from_storage, _load_jobs
from .models import (
    ESCOMapRequest,
    ESCOMapItem,
    ESCOMapBoth,
    JobStatus,
    PolicyReq,
    PolicyRecommendationsResponse,
    UserResultItem,
    AnalysisTitleItem,
    AnalysisRecordItem,
    AnalysisDeleteResult,
    TitleExistsResult,
    AnalysisBatchStatus,
)
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


def _build_user_result_item(job_id: str, info: dict[str, Any], result_type: str, include_content: bool) -> UserResultItem:
    """Shape a completed stored result into a frontend-friendly response item."""
    result_path = info.get("result_path")
    if not result_path:
        raise HTTPException(status_code=500, detail=f"Missing result path for job {job_id}")

    path = Path(result_path)
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"Result file not found for job {job_id}")

    content = load_json(str(path)) if include_content else None
    return UserResultItem(
        job_id=job_id,
        status="done",
        user_id=str(info.get("user_id") or ""),
        result_path=str(path),
        type=result_type,  # type: ignore[arg-type]
        source_job_id=info.get("source_job_id"),
        message=info.get("message"),
        content=content,
    )


def _list_user_results(user_id: str, result_type: str, include_content: bool) -> list[UserResultItem]:
    """Return completed analysis/policy results for a specific user."""
    suffix = ".analysis.json" if result_type == "analysis" else ".policy.json"
    items: list[UserResultItem] = []

    for job_id, info in list_jobs().items():
        if info.get("status") != "done":
            continue
        if info.get("user_id") != user_id:
            continue

        result_path = info.get("result_path")
        if not result_path or not str(result_path).endswith(suffix):
            continue

        items.append(_build_user_result_item(job_id, info, result_type, include_content))

    items.sort(key=lambda item: item.job_id, reverse=True)
    return items

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

    ESCO warm-up (model load + embedding of all ESCO rows) is heavy and, on a
    cold cache, can take minutes. It is run in a background thread so the app
    starts serving requests (e.g. /health) immediately; the first ESCO mapping
    will simply wait until warm-up finishes. The ESCO cache functions are
    idempotent, so a concurrent first request racing the warm-up is safe.
    """
    _load_jobs()
    rehydrate_from_storage(str(STORAGE))

    def _warm() -> None:
        try:
            warm_esco_caches()
            log.info("ESCO caches warmed.")
        except Exception as exc:  # noqa: BLE001 - don't let warm-up kill the app
            log.exception("ESCO cache warm-up failed: %s", exc)

    threading.Thread(target=_warm, name="warm-esco-caches", daemon=True).start()
    log.info("Startup complete (ESCO caches warming in background): storage=%s", STORAGE.resolve())

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
async def analyze_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    query: Optional[str] = None,
    user_id: Optional[str] = Form(default=None),
    title: Optional[str] = Form(default=None),
    sector: Optional[str] = Form(default=None),
    description: Optional[str] = Form(default=None),
) -> JobStatus:
    """
    Enqueue PDF analysis.

    Behavior:
        - Validates .pdf extension and non-empty content
        - Deduplicates by content hash: if a finished job with the same hash and
          title exists, returns its job status immediately
        - Otherwise, stores the PDF and spawns a background task that:
            * processes the PDF
            * writes result JSON (with the title/sector/description embedded)
            * updates job status accordingly

    Metadata:
        - `title`: groups one or more PDF analyses (a title may cover several PDFs)
        - `sector`: high-level sector for the title (one per title)
        - `description`: free-text description for the title (one per title)
        These are persisted in the job registry and embedded in the analysis JSON,
        and are surfaced by the analysis-catalog endpoints below.
    """
    orig_filename = file.filename or ""
    if not orig_filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")

    # Normalize optional metadata (treat blank strings as absent)
    title = (title or "").strip() or None
    sector = (sector or "").strip() or None
    description = (description or "").strip() or None
    orig_filename = orig_filename or None
    created_at = datetime.now(timezone.utc).isoformat()

    file_hash = _sha256(data)

    # Deduplicate by hash if a completed job exists (same user and same title)
    for jid, info in (list_jobs() or {}).items():
        if (
            info.get("status") == "done"
            and info.get("file_hash") == file_hash
            and info.get("result_path")
            and info.get("user_id") == user_id
            and info.get("title") == title
        ):
            return JobStatus(
                job_id=jid,
                status="done",
                message="Duplicate PDF detected. Reusing previous result.",
                result_path=info["result_path"],
                user_id=info.get("user_id"),
                source_job_id=info.get("source_job_id"),
                title=info.get("title"),
                sector=info.get("sector"),
                description=info.get("description"),
                filename=info.get("filename"),
            )

    # New job
    job_id = new_job()
    tmp_path = STORAGE / f"{job_id}.pdf"
    tmp_path.write_bytes(data)

    def _run() -> None:
        """Background task: process the PDF and persist the analysis JSON."""
        try:
            set_status(
                job_id,
                "running",
                message="Processing PDF",
                file_hash=file_hash,
                user_id=user_id,
                title=title,
                sector=sector,
                description=description,
                filename=orig_filename,
                created_at=created_at,
            )
            result: dict[str, Any] = process_pdf(str(tmp_path), query=query)

            # Save the metadata alongside the analysis content
            result["title"] = title
            result["sector"] = sector
            result["description"] = description
            result["filename"] = orig_filename
            result["created_at"] = created_at
            result["user_id"] = user_id

            out_path = STORAGE / f"{job_id}.analysis.json"
            save_json(result, str(out_path))

            set_status(
                job_id,
                "done",
                result_path=str(out_path),
                file_hash=file_hash,
                user_id=user_id,
                title=title,
                sector=sector,
                description=description,
                filename=orig_filename,
                created_at=created_at,
            )
        except Exception as exc:  # noqa: BLE001 - bubble through as status
            log.exception("PDF analysis failed for job %s: %s", job_id, exc)
            set_status(
                job_id,
                "error",
                message=str(exc),
                file_hash=file_hash,
                user_id=user_id,
                title=title,
                sector=sector,
                description=description,
                filename=orig_filename,
                created_at=created_at,
            )
        finally:
            # The raw PDF is only needed for the analysis above; the analysis
            # JSON (and downstream ESCO mapping / recommendations) never re-read
            # it. Delete it to keep the storage volume clean.
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001 - best-effort cleanup
                log.warning("Could not delete temporary PDF for job %s", job_id)

    background_tasks.add_task(_run)
    set_status(
        job_id,
        "queued",
        user_id=user_id,
        title=title,
        sector=sector,
        description=description,
        filename=orig_filename,
        created_at=created_at,
    )
    return JobStatus(
        job_id=job_id,
        status="queued",
        user_id=user_id,
        title=title,
        sector=sector,
        description=description,
        filename=orig_filename,
    )

# -----------------------------------------------------------------------------
# One-click full pipeline: analysis -> ESCO mapping -> policy recommendations
# -----------------------------------------------------------------------------
def _run_full_pipeline(
    job_id: str,
    tmp_path: Path,
    filename: Optional[str],
    user_id: Optional[str],
    title: Optional[str],
    sector: Optional[str],
    description: Optional[str],
    created_at: str,
    query: Optional[str],
) -> None:
    """Analyze one PDF, then generate policy recommendations (with defaults)."""
    common = dict(
        type="analysis", user_id=user_id, title=title, sector=sector,
        description=description, filename=filename, created_at=created_at,
    )
    try:
        set_status(job_id, "running", stage="analyzing", message="Analyzing PDF", **common)
        result: dict[str, Any] = process_pdf(str(tmp_path), query=query)
        result["title"] = title
        result["sector"] = sector
        result["description"] = description
        result["filename"] = filename
        result["created_at"] = created_at
        result["user_id"] = user_id

        out_path = STORAGE / f"{job_id}.analysis.json"
        save_json(result, str(out_path))
        set_status(job_id, "running", stage="recommending",
                   message="Mapping to ESCO and generating recommendations",
                   result_path=str(out_path), **common)

        # Policy recommendations (which also produce the ESCO mapping_evidence).
        techs = result.get("technologies", []) or []
        policy_job_id = str(uuid.uuid4())
        policy_out = STORAGE / f"{policy_job_id}.policy.json"
        set_status(policy_job_id, "running", type="policy",
                   user_id=user_id, source_job_id=job_id)
        try:
            policy = generate_policy_recommendations(
                technologies=techs, target="both",
                similarity_threshold=0.5, max_actions_per_tech=4,
            )
            policy["type"] = "policy"
            policy["user_id"] = user_id
            policy["source_job_id"] = job_id
            policy["created_at"] = datetime.now(timezone.utc).isoformat()
            save_json(policy, str(policy_out))
            set_status(policy_job_id, "done", type="policy",
                       result_path=str(policy_out), user_id=user_id, source_job_id=job_id)
        except Exception as pexc:  # noqa: BLE001 - recommendations are best-effort
            log.exception("Recommendations failed for job %s: %s", job_id, pexc)
            set_status(policy_job_id, "error", type="policy",
                       message=str(pexc), user_id=user_id, source_job_id=job_id)

        set_status(job_id, "done", stage="done", result_path=str(out_path), **common)
    except Exception as exc:  # noqa: BLE001 - surface as job status
        log.exception("Full pipeline failed for job %s: %s", job_id, exc)
        set_status(job_id, "error", stage="error", message=str(exc), **common)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            log.warning("Could not delete temporary PDF for job %s", job_id)


@app.post("/analyze/pdf/full", response_model=AnalysisBatchStatus, tags=["analysis"])
async def analyze_pdf_full(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    user_id: Optional[str] = Form(default=None),
    title: Optional[str] = Form(default=None),
    sector: Optional[str] = Form(default=None),
    description: Optional[str] = Form(default=None),
    query: Optional[str] = Form(default=None),
) -> AnalysisBatchStatus:
    """
    One-click analysis for one or more PDFs.

    Enqueues a background job that, for each PDF and using default parameters,
    runs the whole pipeline — PDF analysis, ESCO mapping and policy
    recommendations — and stores the results. PDFs are processed one at a time.
    Returns immediately with the analysis job ids; poll `/jobs/{job_id}` for the
    per-PDF `stage`, or `/analyses/titles?include_running=true` for aggregate
    progress.
    """
    if not files:
        raise HTTPException(status_code=400, detail="Please upload at least one PDF file.")

    title = (title or "").strip() or None
    sector = (sector or "").strip() or None
    description = (description or "").strip() or None
    created_at = datetime.now(timezone.utc).isoformat()

    prepared: list[tuple[str, Path, Optional[str]]] = []
    for f in files:
        fname = f.filename or ""
        if not fname.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"'{fname}' is not a PDF file.")
        data = await f.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"'{fname}' is empty.")
        job_id = new_job()
        tmp_path = STORAGE / f"{job_id}.pdf"
        tmp_path.write_bytes(data)
        set_status(
            job_id, "queued", stage="queued", type="analysis",
            user_id=user_id, title=title, sector=sector, description=description,
            filename=(fname or None), created_at=created_at, file_hash=_sha256(data),
        )
        prepared.append((job_id, tmp_path, fname or None))

    def _run_all() -> None:
        for job_id, tmp_path, fname in prepared:
            _run_full_pipeline(
                job_id, tmp_path, fname, user_id, title, sector, description, created_at, query
            )

    background_tasks.add_task(_run_all)
    return AnalysisBatchStatus(
        title=title,
        job_ids=[p[0] for p in prepared],
        count=len(prepared),
        status="queued",
    )


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
        filename=Path(path).name
    )


# -----------------------------------------------------------------------------
# Analysis catalog (grouped by title / sector)
# -----------------------------------------------------------------------------
def _is_analysis_job(info: dict[str, Any]) -> bool:
    """True for analysis (not policy) jobs, including ones still running."""
    if info.get("type") == "policy":
        return False
    if info.get("type") == "analysis":
        return True
    # Backward-compat: identify by the result file suffix when type is absent.
    return str(info.get("result_path", "")).endswith(".analysis.json")


def _iter_analysis_jobs(include_running: bool = False):
    """
    Yield (job_id, info) for analysis jobs.

    By default only completed ones; with include_running=True also queued/
    running/error jobs (used to surface in-progress analyses in the catalog).
    """
    for job_id, info in list_jobs().items():
        if not _is_analysis_job(info):
            continue
        if not include_running and info.get("status") != "done":
            continue
        yield job_id, info


def _sort_titles_newest_first(grouped: dict[str, dict[str, Any]]) -> list[str]:
    """Return title keys sorted by created_at descending, then title ascending."""
    keys = sorted(grouped.keys(), key=str.lower)  # stable secondary: title asc
    keys.sort(key=lambda k: grouped[k].get("created_at") or "", reverse=True)  # primary: date desc
    return keys


def _build_analysis_record(job_id: str, info: dict[str, Any], include_content: bool) -> AnalysisRecordItem:
    """
    Shape an analysis job into a catalog record item.

    Tolerates jobs that are still running (no result file yet): the content is
    only loaded when the result file exists and include_content is requested.
    """
    result_path = info.get("result_path") or ""
    content = None
    if result_path and include_content:
        path = Path(result_path)
        if path.exists():
            content = load_json(str(path))

    return AnalysisRecordItem(
        job_id=job_id,
        status=info.get("status", "done"),
        stage=info.get("stage"),
        user_id=(str(info["user_id"]) if info.get("user_id") is not None else None),
        title=info.get("title"),
        sector=info.get("sector"),
        description=info.get("description"),
        filename=info.get("filename"),
        created_at=info.get("created_at"),
        result_path=str(result_path),
        type="analysis",
        source_job_id=info.get("source_job_id"),
        message=info.get("message"),
        content=content,
    )


def _aggregate_title_status(entry: dict[str, Any]) -> None:
    """Compute the aggregate status of a title group from its jobs' statuses."""
    statuses = entry.pop("_statuses", [])
    done = sum(1 for s in statuses if s == "done")
    entry["done_count"] = done
    entry["total"] = len(statuses)
    if any(s in ("pending", "queued", "running") for s in statuses):
        entry["status"] = "running"
    elif any(s == "error" for s in statuses):
        entry["status"] = "done" if done == len(statuses) else "partial"
    else:
        entry["status"] = "done"


@app.get("/analyses/titles", response_model=list[AnalysisTitleItem], tags=["analysis"])
def list_analysis_titles(include_running: bool = Query(default=False)) -> list[AnalysisTitleItem]:
    """
    List every distinct analysis title, with its sector and description.

    A title may cover several PDF analyses; sector and description are
    consistent per title. `count` reports how many analyses share the title.
    Pass `include_running=true` to also include analyses still being processed;
    each item then carries an aggregate `status` (running | done | partial) and
    `done_count`/`total` for a progress indicator.
    """
    grouped: dict[str, dict[str, Any]] = {}
    for job_id, info in _iter_analysis_jobs(include_running):
        title = info.get("title")
        if not title:
            continue
        entry = grouped.setdefault(
            title,
            {"title": title, "sector": None, "description": None, "count": 0,
             "created_at": None, "_statuses": []},
        )
        entry["count"] += 1
        entry["_statuses"].append(info.get("status"))
        # Sector/description are one-per-title; keep the latest non-empty value.
        if info.get("sector"):
            entry["sector"] = info.get("sector")
        if info.get("description"):
            entry["description"] = info.get("description")
        # Represent the title by the most recent analysis date.
        created = info.get("created_at")
        if created and (entry["created_at"] is None or created > entry["created_at"]):
            entry["created_at"] = created

    for entry in grouped.values():
        _aggregate_title_status(entry)

    return [AnalysisTitleItem(**grouped[t]) for t in _sort_titles_newest_first(grouped)]


@app.get("/analyses/title-exists", response_model=TitleExistsResult, tags=["analysis"])
def analysis_title_exists(title: str = Query(..., description="Analysis title to check")) -> TitleExistsResult:
    """
    Check whether an analysis with the given title already exists.

    Comparison is case-insensitive and ignores surrounding whitespace. Use this
    before starting a new analysis so the user can be told to pick a unique
    title (a single analysis may still contain several PDFs under one title).
    """
    norm = (title or "").strip().lower()
    exists = bool(norm) and any(
        (info.get("title") or "").strip().lower() == norm
        for _job_id, info in _iter_analysis_jobs()
    )
    return TitleExistsResult(
        title=title,
        exists=exists,
        message=("An analysis with this title already exists. Please choose a different title."
                 if exists else None),
    )


@app.get("/analyses/by-title/{title}", response_model=list[AnalysisRecordItem], tags=["analysis"])
def list_analyses_by_title(
    title: str,
    include_content: bool = Query(default=False),
    include_running: bool = Query(default=False),
) -> list[AnalysisRecordItem]:
    """
    List the individual PDF analyses under a specific title.

    Pass `include_running=true` to also include PDFs still being processed
    (each record then carries its live `status`/`stage`), so the UI can resume
    polling an analysis that is still in progress.
    """
    items = [
        _build_analysis_record(job_id, info, include_content)
        for job_id, info in _iter_analysis_jobs(include_running)
        if info.get("title") == title
    ]
    items.sort(key=lambda item: item.job_id, reverse=True)
    return items


@app.delete("/analyses/by-title/{title}", response_model=AnalysisDeleteResult, tags=["analysis"])
def delete_analyses_by_title(title: str) -> AnalysisDeleteResult:
    """
    Delete an entire analysis (all PDF analyses that share the given title),
    together with any policy/recommendation results generated from them.

    Removes both the stored result files and the job-registry entries.
    Returns 404 if no analysis with that title exists.
    """
    # Collect the analysis jobs under this title.
    analysis_ids = [
        job_id for job_id, info in _iter_analysis_jobs()
        if info.get("title") == title
    ]
    if not analysis_ids:
        raise HTTPException(status_code=404, detail="No analysis found for this title.")

    analysis_id_set = set(analysis_ids)

    # Collect policy jobs generated from those analyses.
    policy_ids = [
        job_id for job_id, info in list_jobs().items()
        if info.get("source_job_id") in analysis_id_set
        and str(info.get("result_path", "")).endswith(".policy.json")
    ]

    def _remove(job_id: str) -> None:
        """Delete a job's result file, its leftover PDF, and its registry entry."""
        info = get_job(job_id) or {}
        result_path = info.get("result_path")
        if result_path:
            try:
                Path(result_path).unlink(missing_ok=True)
            except Exception:  # noqa: BLE001 - best-effort cleanup
                log.warning("Could not delete result file for job %s", job_id)
        # Remove any leftover source PDF (normally already deleted post-analysis).
        try:
            (STORAGE / f"{job_id}.pdf").unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
        delete_job(job_id)

    for job_id in analysis_ids:
        _remove(job_id)
    for job_id in policy_ids:
        _remove(job_id)

    return AnalysisDeleteResult(
        title=title,
        deleted_analyses=len(analysis_ids),
        deleted_policies=len(policy_ids),
    )


@app.get("/analyses/sectors", response_model=list[str], tags=["analysis"])
def list_analysis_sectors() -> list[str]:
    """List every distinct sector across all analyses."""
    sectors = {
        info.get("sector")
        for _job_id, info in _iter_analysis_jobs()
        if info.get("sector")
    }
    return sorted(sectors, key=str.lower)


@app.get("/analyses/by-sector/{sector}", response_model=list[AnalysisTitleItem], tags=["analysis"])
def list_titles_by_sector(sector: str) -> list[AnalysisTitleItem]:
    """List the distinct titles (with description) that exist for a sector."""
    grouped: dict[str, dict[str, Any]] = {}
    for _job_id, info in _iter_analysis_jobs():
        if info.get("sector") != sector:
            continue
        title = info.get("title")
        if not title:
            continue
        entry = grouped.setdefault(
            title,
            {"title": title, "sector": sector, "description": None, "count": 0, "created_at": None},
        )
        entry["count"] += 1
        if info.get("description"):
            entry["description"] = info.get("description")
        created = info.get("created_at")
        if created and (entry["created_at"] is None or created > entry["created_at"]):
            entry["created_at"] = created

    return [AnalysisTitleItem(**grouped[t]) for t in _sort_titles_newest_first(grouped)]


@app.get("/users/{user_id}/policies", response_model=list[UserResultItem], tags=["policy"])
def list_user_policies(user_id: str, include_content: bool = Query(default=False)) -> list[UserResultItem]:
    """List completed policy recommendation results stored for a specific user."""
    return _list_user_results(user_id, "policy", include_content)


@app.get("/policies/by-title/{title}", response_model=list[UserResultItem], tags=["policy"])
def list_policies_by_title(title: str, include_content: bool = Query(default=False)) -> list[UserResultItem]:
    """
    List the policy/recommendation results generated from the analyses that
    share the given title (regardless of which user generated them).

    A policy job's `source_job_id` points at the analysis job it came from, so
    this returns every completed policy whose source is one of the analyses
    under `title`. Pass `include_content=true` to inline the stored result.
    """
    analysis_ids = {
        job_id for job_id, info in _iter_analysis_jobs()
        if info.get("title") == title
    }
    items: list[UserResultItem] = []
    for job_id, info in list_jobs().items():
        if info.get("status") != "done":
            continue
        if info.get("source_job_id") not in analysis_ids:
            continue
        result_path = info.get("result_path")
        if not result_path or not str(result_path).endswith(".policy.json"):
            continue
        items.append(_build_user_result_item(job_id, info, "policy", include_content))

    items.sort(key=lambda item: item.job_id, reverse=True)
    return items

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
def policy_recommendations(req: PolicyReq, background_tasks: BackgroundTasks):
    """
    Launch an asynchronous job that generates policy recommendations
    for *emerging* technologies.

    Behavior:
        - Resolve technologies from the request or from a completed analysis job.
        - Create a new policy job and run the recommendation generation
          (emerging-tech detection + LLM calls) in the background.
        - Results are saved to `{job_id}.policy.json`.

    Returns:
        Envelope with:
            - job_id
            - result_path
            - emerging_count
            - has_recommendations (bool)
    """
    source_job_id: Optional[str] = None
    techs = req.technologies or []
    if not techs and req.job_id:
        info = get_job(req.job_id)
        if not info or info.get("status") != "done":
            raise HTTPException(status_code=404, detail="Job not found or not done")
        data = load_json(info.get("result_path"))
        techs = data.get("technologies", [])
        source_job_id = req.job_id

    if not techs:
        raise HTTPException(status_code=400, detail="No technologies provided.")

    policy_user_id = req.user_id
    if policy_user_id is None and source_job_id:
        source_info = get_job(source_job_id) or {}
        policy_user_id = source_info.get("user_id")

    policy_job_id = str(uuid.uuid4())
    out_path = STORAGE / f"{policy_job_id}.policy.json"
    set_status(
        policy_job_id,
        "queued",
        type="policy",
        user_id=policy_user_id,
        source_job_id=source_job_id,
    )

    def run_policy_job():
        try:
            set_status(
                policy_job_id,
                "running",
                type="policy",
                user_id=policy_user_id,
                source_job_id=source_job_id,
            )
            result = generate_policy_recommendations(
                technologies=techs,
                target=req.target,
                similarity_threshold=req.similarity_threshold,
                max_actions_per_tech=req.max_actions_per_tech,
                llm_model=req.llm_model,
            )
            # Embed the job linkage so the policy result can be re-associated
            # after a restart even if the registry file is lost.
            result["type"] = "policy"
            result["user_id"] = policy_user_id
            result["source_job_id"] = source_job_id
            result["created_at"] = datetime.now(timezone.utc).isoformat()
            save_json(result, str(out_path))
            set_status(
                policy_job_id,
                "done",
                type="policy",
                result_path=str(out_path),
                user_id=policy_user_id,
                source_job_id=source_job_id,
            )
        except Exception as e:
            set_status(
                policy_job_id,
                "error",
                type="policy",
                message=str(e),
                user_id=policy_user_id,
                source_job_id=source_job_id,
            )

    background_tasks.add_task(run_policy_job)

    return PolicyRecommendationsResponse(
        job_id=policy_job_id,
        result_path=str(out_path),
        emerging_count=0,           # unknown until finished
        has_recommendations=False,  # unknown until finished
    )

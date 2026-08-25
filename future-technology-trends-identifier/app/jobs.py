# -*- coding: utf-8 -*-
"""
In-memory + on-disk job registry.

Public API (stable):
- _load_jobs() -> None
- rehydrate_from_storage(storage_dir: str = "storage") -> None
- new_job() -> str
- set_status(job_id: str, status: str, **kwargs) -> None
- get_job(job_id: str) -> dict | None
- list_jobs() -> dict

Notes:
- Atomic file writes prevent partial/corrupted registry files.
- Registry file: storage/_jobs_registry.json
- Job result files: {storage}/{job_id}.analysis.json

Created on Thu Oct 23 13:33:15 2025

@author: tsoukj
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional
from uuid import uuid4

# Constants
DEFAULT_STORAGE = Path("storage")
JOBS_DB = DEFAULT_STORAGE / "_jobs_registry.json"
ANALYSIS_GLOB = "*.analysis.json"

# Looser "UUID-ish" pattern is fine; keep your behavior
JOB_ID_RE = re.compile(r"([0-9a-fA-F-]{32,})\.analysis\.json$")

# Allowed status values for basic validation
_ALLOWED_STATUSES = {"pending", "queued", "running", "done", "error"}

# In-memory state
_jobs: Dict[str, Dict[str, Any]] = {}
_lock = Lock()

# ---------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------
def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON atomically: tmp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
    tmp.replace(path)


def _save_jobs() -> None:
    """Persist the in-memory registry to disk (best-effort)."""
    try:
        _atomic_write_json(JOBS_DB, _jobs)
    except Exception:
        # Best-effort: avoid crashing callers on I/O issues
        pass

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def _load_jobs() -> None:
    """
    Load jobs from disk into memory (best-effort).
    Called on startup before rehydration.
    """
    try:
        if JOBS_DB.exists():
            with JOBS_DB.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    with _lock:
                        _jobs.clear()
                        _jobs.update(data)
    except Exception:
        # Ignore malformed file; next save will fix it
        pass

def rehydrate_from_storage(storage_dir: str = "storage") -> None:
    """
    Scan storage for existing *.analysis.json and mark them as done.

    Idempotent: safe to call multiple times (e.g., on restart).
    """
    root = Path(storage_dir)
    if not root.exists():
        return

    for p in root.glob(ANALYSIS_GLOB):
        m = JOB_ID_RE.search(p.name)
        if not m:
            continue
        jid = m.group(1)
        with _lock:
            # Preserve any existing metadata; update status & result_path
            meta = _jobs.setdefault(jid, {})
            meta.update({"status": "done", "result_path": str(p)})
    _save_jobs()

def new_job() -> str:
    """
    Create a new job with status 'pending' and return its job_id.
    The caller (e.g., endpoint) can then move it to 'queued' or 'running'.
    """
    jid = str(uuid4())
    with _lock:
        _jobs[jid] = {"status": "pending"}
        _save_jobs()
    return jid

def set_status(job_id: str, status: str, **kwargs: Any) -> None:
    """
    Update job status and arbitrary metadata (message, file_hash, result_path, etc.).

    Unknown status values are allowed but warned against; we still persist for flexibility.
    """
    if status not in _ALLOWED_STATUSES:
        # keep behavior permissive; just coerce obviously empty statuses
        status = status or "pending"

    with _lock:
        meta = _jobs.get(job_id)
        if meta is None:
            # Allow late creation if caller sets status for a previously known job_id
            meta = {}
            _jobs[job_id] = meta
        meta.update({"status": status, **kwargs})
        _save_jobs()

def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Return job metadata or None.

    If the registry is missing but the result file exists on disk,
    a minimal 'done' entry is created and returned.
    """
    with _lock:
        info = _jobs.get(job_id)

    if info:
        return dict(info)  # return a shallow copy to avoid external mutation

    # Lazy fallback: derive from disk if present
    p = DEFAULT_STORAGE / f"{job_id}.analysis.json"
    if p.exists():
        info = {"status": "done", "result_path": str(p)}
        with _lock:
            _jobs[job_id] = info
            _save_jobs()
        return dict(info)

    return None

def list_jobs() -> Dict[str, Dict[str, Any]]:
    """Return a shallow copy of the entire registry (thread-safe snapshot)."""
    with _lock:
        return dict(_jobs)

def delete_job(job_id: str) -> bool:
    """
    Remove a job from the registry and persist the change.

    Returns True if a job was removed, False if the id was not present.
    """
    with _lock:
        existed = _jobs.pop(job_id, None) is not None
        if existed:
            _save_jobs()
    return existed

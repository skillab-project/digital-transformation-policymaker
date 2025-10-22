
from typing import Dict, Any
from uuid import uuid4
from threading import Lock
from pathlib import Path
import json
import re

JOBS_DB = Path("storage") / "_jobs_registry.json"
ANALYSIS_GLOB = "*.analysis.json"
JOB_ID_RE = re.compile(r"([0-9a-fA-F-]{32,})\.analysis\.json$")  # UUID-ish

_jobs: Dict[str, Dict[str, Any]] = {}
_lock = Lock()

def _save_jobs():
    try:
        JOBS_DB.parent.mkdir(exist_ok=True)
        with JOBS_DB.open("w", encoding="utf-8") as f:
            json.dump(_jobs, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

def _load_jobs():
    try:
        if JOBS_DB.exists():
            with JOBS_DB.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    _jobs.update(data)
    except Exception:
        pass

def rehydrate_from_storage(storage_dir: str = "storage"):
    root = Path(storage_dir)
    if not root.exists():
        return
    for p in root.glob(ANALYSIS_GLOB):
        m = JOB_ID_RE.search(p.name)
        if not m:
            continue
        jid = m.group(1)
        with _lock:
            _jobs.setdefault(jid, {})
            _jobs[jid].update({"status": "done", "result_path": str(p)})
    _save_jobs()

def new_job() -> str:
    jid = str(uuid4())
    with _lock:
        _jobs[jid] = {"status": "pending"}
        _save_jobs()
    return jid

def set_status(job_id: str, status: str, **kwargs):
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update({"status": status, **kwargs})
            _save_jobs()

def get_job(job_id: str):
    with _lock:
        info = _jobs.get(job_id)
    if not info:
        p = Path("storage") / f"{job_id}.analysis.json"
        if p.exists():
            info = {"status": "done", "result_path": str(p)}
            with _lock:
                _jobs[job_id] = info
                _save_jobs()
    return info

def list_jobs():
    with _lock:
        return dict(_jobs)

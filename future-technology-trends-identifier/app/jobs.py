
from typing import Dict, Any
from uuid import uuid4
from threading import Lock

_jobs: Dict[str, Dict[str, Any]] = {}
_lock = Lock()

def new_job() -> str:
    jid = str(uuid4())
    with _lock:
        _jobs[jid] = {"status": "pending"}
    return jid

def set_status(job_id: str, status: str, **kwargs):
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update({"status": status, **kwargs})

def get_job(job_id: str):
    with _lock:
        return _jobs.get(job_id)

def list_jobs():
    with _lock:
        return dict(_jobs)

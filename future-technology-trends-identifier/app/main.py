
import pathlib, hashlib
import uuid
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from typing import Optional
from .models import JobStatus, ESCOMapRequest, PolicyReq
from .analyzer import process_pdf, save_json, load_json
from .jobs import new_job, set_status, get_job, list_jobs, rehydrate_from_storage, _load_jobs
from .esco_match import map_technologies_to_esco_occupations, map_technologies_to_esco_skills, map_technologies_to_esco_both, warm_esco_caches
from .policy_recs import generate_policy_recommendations

app = FastAPI(title="Future Tech Trends Analyzer", version="1.0.0")

STORAGE = pathlib.Path("storage")
STORAGE.mkdir(exist_ok=True)

def _sha256(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

@app.on_event("startup")
def _startup():
    _load_jobs()
    rehydrate_from_storage("storage")
    warm_esco_caches()

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/analyze/pdf", response_model=JobStatus)
async def analyze_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...), query: Optional[str] = None):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    file_hash = _sha256(data)

    for jid, info in (list_jobs() or {}).items():
        if info.get("status") == "done" and info.get("file_hash") == file_hash and info.get("result_path"):
            return JobStatus(job_id=jid, status="done", message="Duplicate PDF detected. Reusing previous result.", result_path=info["result_path"])

    job_id = new_job()
    tmp_path = STORAGE / f"{job_id}.pdf"
    with open(tmp_path, "wb") as f:
        f.write(data)

    def run():
        try:
            set_status(job_id, "running", message="Processing PDF", file_hash=file_hash)
            result = process_pdf(str(tmp_path), query=query)
            out_path = STORAGE / f"{job_id}.analysis.json"
            save_json(result, str(out_path))
            set_status(job_id, "done", result_path=str(out_path), file_hash=file_hash)
        except Exception as e:
            set_status(job_id, "error", message=str(e), file_hash=file_hash)

    background_tasks.add_task(run)
    return JobStatus(job_id=job_id, status="queued")

@app.get("/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: str):
    info = get_job(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(job_id=job_id, **info)

@app.get("/results/{job_id}/download")
def download_result(job_id: str):
    info = get_job(job_id)
    if not info or info.get("status") != "done":
        raise HTTPException(status_code=404, detail="Job not found or not done")
    path = info.get("result_path")
    return FileResponse(path, media_type="application/json", filename=f"{job_id}.analysis.json")

@app.post("/map-to-esco")
def map_to_esco(req: ESCOMapRequest):
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
    elif req.target == "skills":
        return map_technologies_to_esco_skills(techs, top_n=req.top_n, threshold=req.threshold)
    else:  # both
        return map_technologies_to_esco_both(techs, top_n=req.top_n, threshold=req.threshold)

@app.post("/policy/recommendations")
def policy_recommendations(req: PolicyReq):
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

    return {
        "job_id": out_job_id,
        "result_path": str(out_path),
        "emerging_count": len(out.get("emerging", [])),
        "has_recommendations": len(out.get("recommendations", {}).get("recommendations", [])) > 0
    }

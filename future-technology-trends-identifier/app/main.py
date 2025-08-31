
import os, json, tempfile, pathlib
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional
from .models import AnalyzeRequest, JobStatus, ESCOMapRequest
from .analyzer import process_pdf, save_json, load_json
from .jobs import new_job, set_status, get_job
from .esco_match import map_technologies_to_esco_occupations, map_technologies_to_esco_skills, map_technologies_to_esco_both

app = FastAPI(title="Future Tech Trends Analyzer", version="1.0.0")

STORAGE = pathlib.Path("storage")
STORAGE.mkdir(exist_ok=True)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/analyze/pdf", response_model=JobStatus)
async def analyze_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...), query: Optional[str] = None):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    job_id = new_job()
    # save file
    tmp_path = STORAGE / f"{job_id}.pdf"
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    def run():
        try:
            set_status(job_id, "running", message="Processing PDF")
            result = process_pdf(str(tmp_path), query=query)
            out_path = STORAGE / f"{job_id}.analysis.json"
            save_json(result, str(out_path))
            set_status(job_id, "done", result_path=str(out_path))
        except Exception as e:
            set_status(job_id, "error", message=str(e))

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

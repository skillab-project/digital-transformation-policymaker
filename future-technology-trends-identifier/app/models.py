from typing import List, Optional, Dict, Literal
from pydantic import BaseModel

class Technology(BaseModel):
    name: str
    description: str
    domain: str
    occupations: List[str]
    confidence: float

class AnalysisResult(BaseModel):
    technologies: List[Technology]

class AnalyzeRequest(BaseModel):
    query: str

class JobStatus(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = None
    result_path: Optional[str] = None

class ESCOMapRequest(BaseModel):
    job_id: Optional[str] = None
    technologies: Optional[List[Technology]] = None
    top_n: int = 5
    threshold: float = 0.5
    target: Literal["occupations", "skills", "both"] = "occupations"

class ESCOItem(BaseModel):
    label: str
    score: float

class ESCOMapItem(BaseModel):
    technology: str
    matches: List[ESCOItem]

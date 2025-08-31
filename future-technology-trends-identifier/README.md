
# Future Technology Trends Identifier

A microservice for analyzing **Future Technology Trends** from unstructured sources (e.g., PDFs of EU policy documents, Horizon Europe work programs) and mapping them to **ESCO Occupations** and **ESCO Skills**. The service is part of the SKILLAB platform.

---

## 🚀 Features

- **PDF Ingestion & Chunking**  
  Upload PDFs; text is chunked with overlap to preserve context.  
- **LLM-based Technology Extraction**  
  Each chunk is processed through an LLM (e.g., Mistral via Ollama/OpenWebUI). Output is structured JSON listing technologies, domains, occupations, and confidence scores.  
- **Job-based Processing**  
  Long analyses run asynchronously. Each request returns a `job_id`, which can be polled until the analysis is complete.  
- **JSON Storage (no zlib)**  
  Results are stored as `.json` files (human-readable).  
- **Mapping to ESCO Occupations & Skills**  
  Extracted technologies can be mapped to both **occupations** (`all_occupations.csv`) and **skills** (`all_skills.csv`), with cosine similarity search over pre-computed embeddings.  
- **Configurable Targets**  
  Mapping supports three modes: `occupations`, `skills`, or `both`.  
- **Caching & Warm-up**  
  ESCO embeddings are cached in memory and preloaded at startup for faster responses.  

---

## 📂 Project Structure

```
app/
 ├── main.py              # FastAPI entrypoint
 ├── analyzer.py          # PDF processing & JSON save/load
 ├── llm_client.py        # Calls to LLM for extraction
 ├── esco_match.py        # Mapping to occupations and skills
 ├── models.py            # Pydantic models
 ├── config.py            # Settings (reads .env)
 └── jobs.py              # Job tracking utils
datasets/
 ├── all_occupations.csv  # ESCO occupations dataset
 └── all_skills.csv       # ESCO skills dataset
storage/
 └── *.analysis.json      # Results of PDF analysis
```

---

## ⚙️ Installation

```bash
git clone <repo>
cd tech-trends-service
pip install -r requirements.txt
```

Environment setup:  
Create `.env` file:
```env
API_URL=http://localhost:3000
API_TOKEN=your_token_here
MODEL_NAME=mistral
PARALLEL_CHUNKS=4
ESCO_OCCUPATIONS_CSV=datasets/all_occupations.csv
ESCO_SKILLS_CSV=datasets/all_skills.csv
```

---

## ▶️ Run the Service

```bash
uvicorn app.main:app --reload
```

---

## 📡 API Endpoints

### 1. Analyze PDF
`POST /analyze/pdf`  
Upload a PDF and start asynchronous analysis.  
Returns a `job_id`.

### 2. Get Job Status
`GET /jobs/{job_id}`  
Check the status of the analysis (`queued`, `running`, `done`).  
When finished, the result is stored as `.json`.

### 3. Map to ESCO
`POST /map-to-esco`  
Map extracted technologies to ESCO entities.

**Request body:**
```json
{
  "job_id": "YOUR_JOB_ID",
  "top_n": 5,
  "threshold": 0.5,
  "target": "both"  // "occupations", "skills", or "both"
}
```

**Response (example):**
```json
{
  "occupations": [
    {"technology": "AI in Education", "matches": [{"label": "University Lecturer", "score": 0.78}]}
  ],
  "skills": [
    {"technology": "AI in Education", "matches": [{"label": "Machine Learning", "score": 0.81}]}
  ]
}
```

---

## 🧪 Testing with Postman

1. **Upload PDF** → `POST /analyze/pdf` with `file` in form-data.  
2. **Check job status** → `GET /jobs/{job_id}` until `"status": "done"`.  
3. **Run mapping** → `POST /map-to-esco` with `job_id` and `target`.  

---

## 📊 Performance Notes

- Embeddings for ESCO occupations & skills are precomputed and cached.  
- Use a lightweight model (`all-MiniLM-L6-v2`) for faster inference.  
- GPU acceleration (CUDA PyTorch) is recommended for large batches.  

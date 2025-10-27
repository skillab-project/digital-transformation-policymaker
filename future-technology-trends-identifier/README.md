
# Future Technology Trends Identifier

A microservice for analyzing **Future Technology Trends** from unstructured sources (e.g., PDFs of EU policy documents, Horizon Europe work programs) and mapping them to **ESCO Occupations** and **ESCO Skills**. The service also identifies **emerging technologies** (not well represented in ESCO) and generates **policy recommendations** for them. It is part of the **SKILLAB** platform.

---

## 🚀 Features

- **PDF Ingestion & Chunking**  
  User upload PDF documents; text is chunked with overlap to preserve context. Duplicate PDFs (same content hash) are automatically detected and re-use previous analysis results.
- **LLM-based Technology Extraction**  
  Each chunk is processed through a local or remote LLM (e.g., Mistral via Ollama/OpenWebUI). Output is structured JSON listing technologies, domains, occupations, and confidence scores.  
- **Job-based Processing**  
  Long analyses run asynchronously. Each request returns a `job_id`, which can be polled until the analysis is complete.  
- **JSON Storage**  
  Results are stored as a human-readable `.analysis.json` or `.policy.json` files in `/storage`.  
- **Mapping to ESCO Occupations & Skills**  
  Extracted technologies are semantically mapped to **ESCO occupations** and **skills**, with cosine similarity search over pre-computed **SentenceTransformer** embeddings.
- **Caching & Warm-up**  
  ESCO embeddings are cached in memory and preloaded at startup for faster responses.  
- **Configurable Mapping Targets**  
  Mapping supports three modes: `occupations`, `skills`, or `both`.  
- **Emerging Technology Detection**  
  A technology is flagged emerging if it has no ESCO matches above the threshold (configurable per request), or either its skill or occupation match list is empty.  
- **Policy Recommendation Pipeline**  
  For emerging technologies, the service generates structured, machine-readable policy actions (training, funding, KPIs, incentives, etc.).

---

## 📂 Project Structure

```
app/
 ├── main.py              # FastAPI entrypoint
 ├── analyzer.py          # PDF processing & JSON save/load
 ├── pdf_processor.py     # Extraction, cleaning, section detection
 ├── llm_client.py        # Calls to LLM for extraction
 ├── esco_match.py        # Mapping to occupations and skills
 ├── policy_recs.py       # Emerging-tech detection + policy generation
 ├── models.py            # Pydantic models
 ├── config.py            # Settings (reads .env)
 └── jobs.py              # Job tracking utils
datasets/
 ├── all_occupations.csv  # ESCO occupations dataset
 └── all_skills.csv       # ESCO skills dataset
storage/
 ├── *.analysis.json      # PDF analysis results
 ├── *.policy.json        # Policy recommendation results
 └── esco_cache/          # Cached ESCO embeddings (.parquet + .npy)
```

---

## ⚙️ Installation (Local)

```bash
git clone <repo>
cd future-technology-trends-identifier
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

## 🐳 Docker Deployment

**Build image**
```bash
docker build -t skillab-future-tech:latest .
```

**Run container**
```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/datasets:/app/datasets" \
  -v "$(pwd)/storage:/app/storage" \
  -e API_TOKEN="your_token_here" \
  -e API_URL="http://160.40.52.27:3000" \
  skillab-future-tech:latest
```

---

## ▶️ Run the Service

```bash
uvicorn app.main:app --reload
```

**Startup behavior**

- Rehydrates previous job metadata
- Pre-loads cached ESCO embeddings
- Ready for immediate requests

---

## 📡 API Endpoints

### 1. Analyze PDF
`POST /analyze/pdf`  
Upload a PDF and start asynchronous analysis.

**Response:**
```json
{
  "job_id": "abc123",
  "status": "queued"
}
```

### 2. Get Job Status
`GET /jobs/{job_id}`  
Check the status of the analysis (`queued`, `running`, `done`, or `error`).  
When finished, the result is stored as `.json`.

### 3. Map to ESCO
`POST /map-to-esco`  
Map extracted technologies to ESCO occupations and/or skills.

**Request:**
```json
{
  "job_id": {job_id},
  "top_n": 5,
  "threshold": 0.5,
  "target": "both"  // "occupations", "skills", or "both"
}
```

**Response:**
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

### 4. Policy Recommendations
`POST /policy/recommendations`
Generates policy recommendations for emerging technologies (detected automatically).

```json
{
  "job_id": {job_id},
  "target": "both",
  "similarity_threshold": 0.5
}
```

**Response:**
```json
{
  "job_id": "abc123",
  "result_path": "storage/abc123.policy.json",
  "emerging_count": 3,
  "has_recommendations": true
}
```

**Optional:**
Add `?include_content=true` to return full recommendations inline:

`POST /policy/recommendations?include_content=true`

**Response (with inline content):**
```json
{
  "job_id": "abc123",
  "result_path": "storage/abc123.policy.json",
  "emerging_count": 3,
  "has_recommendations": true,
  "content": {
    "emerging": [...],
    "recommendations": [
      {
        "technology": "Quantum Networking",
        "actions": [
          {"area": "Training/Reskilling", "action": "Create pilot curricula", "priority": "High"}
        ]
      }
    ],
    "mapping_evidence": {...}
  }
}

---

## 🧪 Example Workflow

1. **Upload a Horizon Europe PDF** → `/analyze/pdf`
2. **Check job status** → `/jobs/{job_id}`
3. **Map extracted technologies** → `/map-to-esco`
4. **Generate policy recommendations** → `/policy/recommendations`

---

## 🧭 API Documentation
After starting the service, the interactive Swagger UI is available at:

👉 [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📊 Performance Notes

- Embeddings for ESCO occupations & skills are precomputed and cached.  
- Use a lightweight model (`all-MiniLM-L6-v2`) for faster inference.  
- GPU acceleration (CUDA PyTorch) is recommended for large batches.  

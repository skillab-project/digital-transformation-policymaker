
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
  Long analyses run asynchronously. Each request is user-linked via an optional `user_id` and returns a `job_id`, which can be polled until the analysis is complete.  
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
esco_data/
 ├── all_occupations.csv  # ESCO occupations dataset
 └── all_skills.csv       # ESCO skills dataset
storage/
 ├── _jobs_registry.json  # Persisted job metadata
 ├── *.pdf                # Uploaded source PDFs
 ├── *.analysis.json      # PDF analysis results
 ├── *.policy.json        # Policy recommendation results
 └── esco_cache/          # Cached ESCO embeddings (.parquet + .npy)
tests/
 └── test_api.py          # API tests
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
MODEL_NAME=mistral:latest
PARALLEL_CHUNKS=4
ESCO_OCCUPATIONS_CSV=esco_data/all_occupations.csv
ESCO_SKILLS_CSV=esco_data/all_skills.csv
```

---

## 🐳 Docker Deployment

**Build image**
```bash
docker build -t future-technology-trends-identifier .
```

**Run container**
```bash
docker run -p 8000:8000 future-technology-trends-identifier
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

## Storage Model

This service uses file-based persistence, not a database.

Stored on disk:

- Uploaded PDFs in `storage/{job_id}.pdf`
- Analysis results in `storage/{job_id}.analysis.json`
- Policy results in `storage/{job_id}.policy.json`
- Job metadata in `storage/_jobs_registry.json`
- ESCO embedding caches in `storage/esco_cache/`

Job metadata may include:

- `status`
- `result_path`
- `message`
- `file_hash`
- `user_id`
- `source_job_id`
- `type`

---

## 📡 API Endpoints

### 1. Analyze PDF

`POST /analyze/pdf`

Upload a PDF as multipart form-data and start asynchronous analysis.

**Form fields:**

- `file`: required PDF file
- `user_id`: optional frontend/user identifier

**Example response:**

```json
{
  "job_id": "abc123",
  "status": "queued",
  "message": null,
  "result_path": null,
  "user_id": "user-123",
  "source_job_id": null,
  "type": null
}
```

**Notes:**

- Duplicate PDF reuse is scoped to the same `user_id`
- If the same user uploads the same PDF again, the previous completed analysis is reused

### 2. Get Job Status

`GET /jobs/{job_id}`

Check the status of the analysis (`queued`, `running`, `done`, or `error`).

**Example response:**

```json
{
  "job_id": "abc123",
  "status": "done",
  "message": null,
  "result_path": "storage/abc123.analysis.json",
  "user_id": "user-123",
  "source_job_id": null,
  "type": null
}
```

### 3. Download Stored Result

`GET /results/{job_id}/download`

Downloads the stored JSON result (`*.analysis.json` or `*.policy.json`) for a completed analysis or policy job.

### 4. Map to ESCO

`POST /map-to-esco`

Maps technologies to ESCO occupations, skills, or both.

**Example request:**

```json
{
  "job_id": "abc123",
  "top_n": 5,
  "threshold": 0.5,
  "target": "both"
}
```

**Example response:**

```json
{
  "occupations": [
    {
      "technology": "AI in Education",
      "matches": [
        {
          "label": "University Lecturer",
          "score": 0.78
        }
      ]
    }
  ],
  "skills": [
    {
      "technology": "AI in Education",
      "matches": [
        {
          "label": "Machine Learning",
          "score": 0.81
        }
      ]
    }
  ]
}
```

### 5. Generate Policy Recommendations

`POST /policy/recommendations`

Queues a background job that:

1. Detects emerging technologies
2. Calls the LLM for each emerging technology
3. Saves structured policy recommendations to `storage/{policy_job_id}.policy.json`

**Example request:**

```json
{
  "job_id": "abc123",
  "user_id": "user-123",
  "target": "both",
  "similarity_threshold": 0.5
}
```

**Example response:**

```json
{
  "job_id": "policy-job-id",
  "result_path": "storage/policy-job-id.policy.json",
  "emerging_count": 3,
  "has_recommendations": true
}
```

**Notes:**

- If `job_id` is provided, the policy job inherits the source analysis job's `user_id` when `user_id` is omitted
- Policy jobs store `source_job_id` so the frontend can link them back to the originating analysis

### 6. List Analysis Results for a User

`GET /users/{user_id}/analyses`

Returns completed analysis results for a specific user.

**Query params:**

- `include_content=false` by default
- Set `include_content=true` to include the parsed `.analysis.json` content inline

**Example response:**

```json
[
  {
    "job_id": "abc123",
    "status": "done",
    "user_id": "user-123",
    "result_path": "storage/abc123.analysis.json",
    "type": "analysis",
    "source_job_id": null,
    "message": null,
    "content": null
  }
]
```

### 7. List Policy Results for a User

`GET /users/{user_id}/policies`

Returns completed policy recommendation results for a specific user.

**Query params:**

- `include_content=false` by default
- Set `include_content=true` to include the parsed `.policy.json` content inline

**Example response:**

```json
[
  {
    "job_id": "policy-job-id",
    "status": "done",
    "user_id": "user-123",
    "result_path": "storage/policy-job-id.policy.json",
    "type": "policy",
    "source_job_id": "abc123",
    "message": null,
    "content": {
      "emerging": [],
      "recommendations": [],
      "mapping_evidence": {}
    }
  }
]
```

---

## 🧪 Example Workflow

1. Upload a Horizon Europe PDF to `POST /analyze/pdf` with optional `user_id`
2. Poll `GET /jobs/{job_id}` until analysis is complete
3. Map technologies with `POST /map-to-esco`
4. Launch policy generation with `POST /policy/recommendations`
5. Poll `GET /jobs/{policy_job_id}` until policy generation is complete
6. Fetch all stored analyses from `GET /users/{user_id}/analyses`
7. Fetch all stored policy results from `GET /users/{user_id}/policies`
8. Download any specific JSON result with `GET /results/{job_id}/download`

---

## 🧭 API Documentation

After starting the service, Swagger UI is available at:

👉 [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📊 Performance Notes

- ESCO occupation and skill embeddings are precomputed and cached
- `all-MiniLM-L6-v2` is the default lightweight embedding model
- GPU acceleration is recommended for larger workloads

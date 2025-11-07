
# Policy Success Evaluator

A microservice for evaluating the **success of policies and KPIs** within the SKILLAB platform. It analyses KPI trajectories (current vs. target values, time series trends, and gaps) and generates **policy recommendations** aimed at improving or consolidating performance.

---

## 🚀 Features

- **KPI Input & Trend Analysis**  
  Accepts one or more KPIs along with sector, region, and policy context.  
  Automatically computes linear trend, projected time to target, and on/off-track status.

- **LLM-Driven Recommendation Engine**  
  Uses a Large Language Model (e.g., Mistral via OpenWebUI or Ollama) to generate actionable, structured policy recommendations based on KPI progress and EU policy priorities.

- **On-Track vs. Off-Track Adaptation**  
  Dynamically adjusts lever types and intervention intensity depending on whether a KPI is improving or stagnating.  
  On-track KPIs → consolidation and monitoring.  
  Off-track KPIs → accelerators (funding, training, regulation, etc.).
 
 ---
 
 ## 📂 Project Structure

```
├── app/
│   └── main.py             # FastAPI entrypoint and main logic
├── .env                    # Environment configuration (LLM endpoint, model, etc.)
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container build file
└── README.md               # Documentation

```

---

## ⚙️ Installation (Local)

```bash
git clone <repo>
cd policy-success-evaluator
pip install -r requirements.txt
```

Environment setup:  
Create `.env` file:
```env
API_URL=http://localhost:3000
API_TOKEN=your_token_here
MODEL_NAME=mistral:latest
TEMPERATURE=0.1
SEED=42
TIMEOUT=60
```

---

## 🐳 Docker Deployment

**Build image**
```bash
docker build -t policy-success-evaluator .
```

**Run container**
```bash
docker run -p 8000:8000 --env-file .env policy-success-evaluator
```

---

## ▶️ Run the Service

```bash
uvicorn main:app --reload --port 8000
```

---

## 📡 API Endpoints

### 1. Generate KPI Recommendations
`POST /kpi/recommendations`  
Evaluates one or more KPIs and generates tailored policy recommendations.

**Request:**
```json
{
  "kpis": [
    {
      "id": "kpi_digital_adoption",
      "name": "SME Digital Adoption Rate",
      "unit": "percentage",
      "direction": "higher_is_better",
      "current_value": 42,
      "target_value": 60,
      "target_deadline": "2026-Q4",
      "time_series": [
        {"period": "2024-Q1", "value": 51.5},
        {"period": "2024-Q2", "value": 53.0},
        {"period": "2024-Q3", "value": 54.0}
      ]
    }
  ],
  "scope": {
    "sector": "SMEs",
    "region": "Central Macedonia, Greece",
    "policy": "Digital Boost 2024"
  }
}

```

**Response:**
```json
[
  {
    "kpi_id": "kpi_digital_adoption",
    "trend_analysis": "Increasing at +1.15 per quarter. On current pace, target met in ~16 quarters. This is off track relative to deadline 2026-Q4.",
    "recommendations": [
      {
        "lever_type": "Training",
        "title": "Digital Skills Training Program for SMEs",
        "mechanism": "Provide digital skills training programs tailored to the needs of SMEs in Central Macedonia, Greece, covering topics such as e-commerce, online marketing, and digital productivity tools.",
        "rational": "To equip SMEs with the necessary digital skills to improve their digital adoption rates.",
        "expected_impact": "Medium",
        "time_to_effect": "Medium",
        "risks_tradeoffs": "Ensure programs are relevant and engaging for SME owners and employees; adequate resources must be allocated for program development and delivery.",
        "prerequisites": [
		  "Identify training needs of SMEs in Central Macedonia, Greece",
		  "Secure partnerships with local digital skills trainers"
		]
      },
	  {
        "lever_type": "Advisory Support",
        "title": "Digital Adoption Advisory Service for SMEs",
        "mechanism": "Offer free, one-on-one consultations to SMEs in Central Macedonia, providing advice on digital tools and strategies tailored to their specific needs.",
        "rational": "To provide personalized guidance to SMEs on how to improve their digital adoption rates through targeted advice and recommendations.",
        "expected_impact": "Medium",
        "time_to_effect": "Short",
        "risks_tradeoffs": "Ensure advisors have sufficient digital expertise and knowledge of the SME sector; manage high demand for consultations.",
        "prerequisites": [
          "Recruit digital experts with knowledge of the SME sector",
          "Establish an appointment booking system"
        ]
      }
    ]
  }
]
```

### 2. Health Check
`GET /health`  
Returns service status and active model.

**Response:**
```json
{"status": "ok", "model": "mistral:latest"}
```

---

## 🧪 Example Workflow

1. **Send KPIs & context** → `/kpi/recommendations`
2. **Service calculates trends** and identifies if each KPI is on or off track
3. **LLM generates structured recommendations** according to policy type
4. **Response returned as JSON**

---

## 🧭 API Documentation
After starting the service, the interactive Swagger UI is available at:

👉 [http://localhost:8000/docs](http://localhost:8000/docs)

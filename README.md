# AutoFlow AI 🚗

> Vehicle sensor anomaly detection pipeline — FastAPI · scikit-learn · Docker · AWS Lambda + API Gateway (CDK)

AutoFlow AI is a production-style REST API that ingests automotive sensor telemetry and runs it through an **Isolation Forest** machine learning pipeline to detect anomalous vehicle behaviour in real time. The service is containerised with Docker and deployed to AWS Lambda via API Gateway using **AWS CDK** (Infrastructure-as-Code).

---

## Architecture

```
HTTP Client
    │
    ▼
API Gateway  (AWS / LocalStack)
    │
    ▼
Lambda Function  ←── Mangum ASGI adapter
    │
    ▼
FastAPI + Pydantic
    │
    ▼
Isolation Forest Pipeline  (scikit-learn)
    │
    ▼
JSON Response  { anomaly_score, severity, flagged_sensors }
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI + Pydantic v2 |
| ML pipeline | scikit-learn (Isolation Forest) |
| ASGI → Lambda | Mangum |
| Containerisation | Docker (multi-stage build) |
| Infrastructure-as-Code | AWS CDK v2 (Python) |
| Local AWS | LocalStack |
| Testing | pytest + pytest-asyncio |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Service info |
| GET | `/api/v1/health` | Health check + pipeline status |
| GET | `/api/v1/pipeline/status` | Model metadata |
| POST | `/api/v1/analyze` | **Run anomaly detection pipeline** |

### Example Request

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "readings": [
      {
        "vehicle_id": "VIN-AU-20240001",
        "speed_kmh": 95.0,
        "engine_rpm": 2800.0,
        "engine_temp_c": 92.0,
        "battery_voltage_v": 13.4,
        "fuel_level_pct": 65.0,
        "oil_pressure_bar": 3.2
      }
    ],
    "sensitivity": 0.05
  }'
```

### Example Response

```json
{
  "total_readings": 1,
  "anomalies_detected": 0,
  "anomaly_rate_pct": 0.0,
  "results": [
    {
      "vehicle_id": "VIN-AU-20240001",
      "is_anomaly": false,
      "anomaly_score": 0.082341,
      "anomaly_probability": 0.118,
      "flagged_sensors": [],
      "severity": "normal"
    }
  ],
  "pipeline_version": "1.0.0",
  "model_type": "IsolationForest"
}
```

---

## Run Locally (no Docker)

```bash
# 1. Clone and enter project
git clone https://github.com/J4jatin/autoflow-ai
cd autoflow-ai

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Start the API
python app/main.py

# 5. Open docs → http://localhost:8000/docs
```

---

## Run with Docker

```bash
# Build and start
docker-compose up --build autoflow-api

# API available at http://localhost:8000/docs
```

---

## Run Tests

```bash
pytest tests/ -v --cov=app --cov-report=term-missing
```

Expected output:
```
tests/test_api.py::test_health_check              PASSED
tests/test_api.py::test_analyze_normal_reading    PASSED
tests/test_api.py::test_analyze_anomalous_reading PASSED
tests/test_api.py::test_analyze_batch             PASSED
tests/test_pipeline.py::test_pipeline_trains      PASSED
tests/test_pipeline.py::test_predict_normal       PASSED
tests/test_pipeline.py::test_predict_anomalous    PASSED
...
```

---

## Deploy to LocalStack (local AWS)

```bash
# Step 1 — Start LocalStack
docker-compose up -d localstack

# Step 2 — Install CDK tools
npm install -g aws-cdk aws-cdk-local

# Step 3 — Install CDK Python deps
cd infrastructure
pip install -r requirements.txt

# Step 4 — Bootstrap + Deploy
cdklocal bootstrap
cdklocal deploy

# Step 5 — Note the API URL from outputs and test it
curl http://localhost:4566/restapis/.../prod/api/v1/health
```

---

## Project Structure

```
autoflow-ai/
├── app/
│   ├── main.py              # FastAPI app + Lambda handler
│   ├── models.py            # Pydantic request/response schemas
│   ├── api/
│   │   └── routes.py        # API endpoint definitions
│   └── pipeline/
│       └── detector.py      # Isolation Forest ML pipeline
├── tests/
│   ├── conftest.py          # Fixtures
│   ├── test_api.py          # Integration tests
│   └── test_pipeline.py     # Unit tests
├── infrastructure/
│   ├── app.py               # CDK app entry point
│   └── autoflow_stack.py    # Lambda + API Gateway stack
├── Dockerfile               # Multi-stage Docker build
├── docker-compose.yml       # API + LocalStack services
└── requirements.txt
```

---

## Author

**Jattin Shah** — M.Sc. Computational Modeling and Simulation, TU Dresden  
GitHub: [github.com/J4jatin](https://github.com/J4jatin)

# API Module

> FastAPI REST API for running analyses programmatically.

## Overview

The `api.py` module provides REST endpoints for all analysis functions, enabling integration with other systems and automation.

## Starting the Server

```bash
# Development
uvicorn src.api:app --reload --port 8000

# Production
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4

# Docker
docker-compose up api
```

---

## Endpoints

### `GET /` - API Info
Returns API version and available endpoints.

```bash
curl http://localhost:8000/
```

### `GET /health` - Health Check
Check if service is running.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{"status": "healthy", "timestamp": "2024-03-15T10:30:00"}
```

---

### `POST /analyze` - Run Analysis

Run causal impact analysis.

**Request:**
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "intervention_date": "2024-03-01",
    "segment_type": "channel",
    "segment_value": "email",
    "alpha": 0.05
  }'
```

**Response:**
```json
{
  "segment": "email",
  "cumulative_effect": 42137.64,
  "average_effect": 1361.53,
  "relative_effect": 0.35,
  "p_value": 0.001,
  "significant": true,
  "ci_lower": 38521.10,
  "ci_upper": 45754.18
}
```

---

### `GET /segments` - List Segments

Get available segments for analysis.

```bash
curl http://localhost:8000/segments
```

**Response:**
```json
{
  "channels": ["email", "social", "search", "display"],
  "countries": ["US", "UK", "CA", "DE"],
  "devices": ["mobile", "desktop", "tablet"]
}
```

---

### `GET /segments/{type}` - Batch Analysis

Run analysis for all values of a segment type.

```bash
curl http://localhost:8000/segments/channel
```

---

## Python Client

```python
import requests

# Run analysis
response = requests.post(
    "http://localhost:8000/analyze",
    json={
        "intervention_date": "2024-03-01",
        "segment_type": "channel",
        "segment_value": "email"
    }
)

result = response.json()
print(f"Effect: ${result['cumulative_effect']:,.2f}")
```

---

## OpenAPI Docs

Interactive documentation available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## See Also

- [Database Module](database.md) - Result persistence
- [Scheduler](scheduler.md) - Automated runs
- [Deployment Guide](../deployment.md) - Production setup

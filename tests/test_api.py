"""
Integration tests for AutoFlow AI API endpoints.
Tests the full request → pipeline → response flow.
"""
import pytest


def test_health_check(client):
    """API health endpoint returns ok and pipeline_ready=True."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["pipeline_ready"] is True
    assert data["model_trained"] is True


def test_root_endpoint(client):
    """Root endpoint returns service metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "AutoFlow AI"
    assert "docs" in data


def test_pipeline_status(client):
    """Pipeline status endpoint returns model metadata."""
    response = client.get("/api/v1/pipeline/status")
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "IsolationForest"
    assert data["trained"] is True
    assert "feature_columns" in data


def test_analyze_normal_reading(client, normal_reading):
    """A normal sensor reading should return severity=normal or low."""
    payload = {"readings": [normal_reading], "sensitivity": 0.05}
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_readings"] == 1
    result = data["results"][0]
    assert result["vehicle_id"] == "VIN-TEST-001"
    assert result["severity"] in ("normal", "low")
    assert 0.0 <= result["anomaly_probability"] <= 1.0


def test_analyze_anomalous_reading(client, anomalous_reading):
    """An extreme sensor reading should be flagged as anomalous."""
    payload = {"readings": [anomalous_reading], "sensitivity": 0.05}
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    result = data["results"][0]
    assert result["is_anomaly"] is True
    assert len(result["flagged_sensors"]) > 0
    assert result["severity"] in ("medium", "high", "critical")


def test_analyze_batch(client, normal_reading, anomalous_reading):
    """Batch request with mixed readings returns correct totals."""
    payload = {
        "readings": [normal_reading, anomalous_reading],
        "sensitivity": 0.05,
    }
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_readings"] == 2
    assert 0 <= data["anomalies_detected"] <= 2
    assert len(data["results"]) == 2


def test_analyze_invalid_speed(client, normal_reading):
    """Speed above 400 km/h should fail Pydantic validation."""
    bad = {**normal_reading, "speed_kmh": 999.0}
    payload = {"readings": [bad]}
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 422  # Unprocessable Entity


def test_analyze_empty_readings(client):
    """Empty readings list should fail validation."""
    payload = {"readings": []}
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 422

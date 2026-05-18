"""
Unit tests for the AnomalyDetectionPipeline.
Tests the ML pipeline directly, independent of FastAPI.
"""
import pytest
from app.pipeline.detector import AnomalyDetectionPipeline, FEATURE_COLUMNS


@pytest.fixture(scope="module")
def trained_pipeline():
    p = AnomalyDetectionPipeline()
    p.train()
    return p


def test_pipeline_trains(trained_pipeline):
    """Pipeline should be marked as trained after .train()."""
    assert trained_pipeline.is_trained is True
    assert trained_pipeline.model is not None
    assert trained_pipeline.scaler is not None


def test_predict_normal(trained_pipeline):
    """Normal reading should produce low anomaly probability."""
    reading = {
        "vehicle_id": "VIN-UNIT-001",
        "speed_kmh": 80.0,
        "engine_rpm": 2200.0,
        "engine_temp_c": 88.0,
        "battery_voltage_v": 13.4,
        "fuel_level_pct": 55.0,
        "oil_pressure_bar": 3.1,
    }
    results = trained_pipeline.predict([reading])
    assert len(results) == 1
    r = results[0]
    assert r["vehicle_id"] == "VIN-UNIT-001"
    assert r["severity"] in ("normal", "low")
    assert r["anomaly_probability"] < 0.7


def test_predict_anomalous(trained_pipeline):
    """Extreme reading should produce high anomaly probability."""
    reading = {
        "vehicle_id": "VIN-UNIT-FAULT",
        "speed_kmh": 300.0,
        "engine_rpm": 8800.0,
        "engine_temp_c": 148.0,
        "battery_voltage_v": 8.0,
        "fuel_level_pct": 1.0,
        "oil_pressure_bar": 0.2,
    }
    results = trained_pipeline.predict([reading])
    r = results[0]
    assert r["is_anomaly"] is True
    assert len(r["flagged_sensors"]) >= 3
    assert r["severity"] in ("high", "critical")


def test_predict_batch(trained_pipeline):
    """Batch prediction returns one result per reading."""
    readings = [
        {
            "vehicle_id": f"VIN-BATCH-{i:03d}",
            "speed_kmh": 70.0 + i,
            "engine_rpm": 2000.0 + i * 10,
            "engine_temp_c": 88.0,
            "battery_voltage_v": 13.2,
            "fuel_level_pct": 50.0,
            "oil_pressure_bar": 3.0,
        }
        for i in range(10)
    ]
    results = trained_pipeline.predict(readings)
    assert len(results) == 10
    for r in results:
        assert "anomaly_score" in r
        assert "severity" in r


def test_predict_requires_training():
    """Calling predict on untrained pipeline should raise RuntimeError."""
    p = AnomalyDetectionPipeline()
    with pytest.raises(RuntimeError, match="Pipeline not trained"):
        p.predict([{"vehicle_id": "x", "speed_kmh": 80, "engine_rpm": 2000,
                    "engine_temp_c": 90, "battery_voltage_v": 13,
                    "fuel_level_pct": 50, "oil_pressure_bar": 3}])


def test_flag_sensors(trained_pipeline):
    """Out-of-range sensors should be correctly flagged."""
    reading = {
        "vehicle_id": "VIN-FLAG-001",
        "speed_kmh": 250.0,   # out of range
        "engine_rpm": 2500.0,
        "engine_temp_c": 90.0,
        "battery_voltage_v": 8.0,  # out of range
        "fuel_level_pct": 50.0,
        "oil_pressure_bar": 3.0,
    }
    flagged = trained_pipeline._flag_sensors(reading)
    assert "speed_kmh" in flagged
    assert "battery_voltage_v" in flagged
    assert "engine_rpm" not in flagged

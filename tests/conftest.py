"""
Pytest configuration and shared fixtures for AutoFlow AI tests.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.pipeline.detector import pipeline


@pytest.fixture(scope="session", autouse=True)
def train_pipeline():
    """Train the pipeline once before all tests."""
    if not pipeline.is_trained:
        pipeline.train()


@pytest.fixture(scope="session")
def client(train_pipeline):
    """Shared FastAPI test client."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def normal_reading():
    return {
        "vehicle_id": "VIN-TEST-001",
        "speed_kmh": 90.0,
        "engine_rpm": 2500.0,
        "engine_temp_c": 90.0,
        "battery_voltage_v": 13.2,
        "fuel_level_pct": 60.0,
        "oil_pressure_bar": 3.0,
    }


@pytest.fixture
def anomalous_reading():
    return {
        "vehicle_id": "VIN-TEST-ANOMALY",
        "speed_kmh": 250.0,       # way too fast
        "engine_rpm": 8500.0,     # over redline
        "engine_temp_c": 145.0,   # overheating
        "battery_voltage_v": 9.0, # critically low
        "fuel_level_pct": 2.0,    # almost empty
        "oil_pressure_bar": 0.5,  # dangerously low
    }

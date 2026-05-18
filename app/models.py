"""
Pydantic models for AutoFlow AI API.
Defines request/response schemas with full validation.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


class SensorReading(BaseModel):
    """A single automotive sensor data point."""

    vehicle_id: str = Field(..., description="Unique vehicle identifier", json_schema_extra={"example": "VIN-AU-20240001"})
    speed_kmh: float = Field(..., ge=0, le=400, description="Vehicle speed in km/h")
    engine_rpm: float = Field(..., ge=0, le=9000, description="Engine RPM")
    engine_temp_c: float = Field(..., ge=-40, le=150, description="Engine temperature in Celsius")
    battery_voltage_v: float = Field(..., ge=0, le=20, description="Battery voltage in Volts")
    fuel_level_pct: float = Field(..., ge=0, le=100, description="Fuel level as percentage")
    oil_pressure_bar: float = Field(..., ge=0, le=10, description="Oil pressure in bar")
    timestamp_ms: Optional[int] = Field(None, description="Unix timestamp in milliseconds")

    @field_validator("vehicle_id")
    @classmethod
    def vehicle_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("vehicle_id must not be empty")
        return v.strip()


class BatchSensorRequest(BaseModel):
    """Batch of sensor readings for pipeline processing."""

    readings: list[SensorReading] = Field(
        ..., min_length=1, max_length=500, description="List of sensor readings to analyze"
    )
    sensitivity: float = Field(
        default=0.05,
        ge=0.01,
        le=0.5,
        description="Anomaly sensitivity (contamination rate). Higher = more anomalies flagged.",
    )


class AnomalyResult(BaseModel):
    """Anomaly detection result for a single sensor reading."""

    vehicle_id: str
    is_anomaly: bool
    anomaly_score: float = Field(..., description="Raw isolation forest score. More negative = more anomalous.")
    anomaly_probability: float = Field(..., ge=0, le=1, description="Normalised anomaly probability [0, 1]")
    flagged_sensors: list[str] = Field(default_factory=list, description="Sensors that deviate from normal range")
    severity: str = Field(..., description="Severity level: normal | low | medium | high | critical")


class BatchAnalysisResponse(BaseModel):
    """Full pipeline response for a batch analysis request."""

    total_readings: int
    anomalies_detected: int
    anomaly_rate_pct: float
    results: list[AnomalyResult]
    pipeline_version: str = "1.0.0"
    model_type: str = "IsolationForest"


class HealthResponse(BaseModel):
    """API health check response."""

    status: str = "ok"
    pipeline_ready: bool
    model_trained: bool
    version: str = "1.0.0"

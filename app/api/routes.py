"""
AutoFlow AI — API Routes
All endpoints for the vehicle sensor anomaly detection service.
"""
from fastapi import APIRouter, HTTPException, status
from app.models import (
    BatchSensorRequest,
    BatchAnalysisResponse,
    AnomalyResult,
    HealthResponse,
)
from app.pipeline.detector import pipeline
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API and pipeline status."""
    return HealthResponse(
        status="ok",
        pipeline_ready=pipeline.is_trained,
        model_trained=pipeline.is_trained,
    )


@router.post(
    "/analyze",
    response_model=BatchAnalysisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Pipeline"],
    summary="Analyze vehicle sensor data for anomalies",
    description=(
        "Submit a batch of vehicle sensor readings. "
        "The AI pipeline runs Isolation Forest anomaly detection "
        "and returns per-vehicle results with severity scores."
    ),
)
async def analyze_sensors(request: BatchSensorRequest):
    """
    Run the anomaly detection pipeline on a batch of sensor readings.

    - Accepts 1–500 readings per request
    - Returns anomaly score, probability, flagged sensors, and severity
    - Severity levels: normal | low | medium | high | critical
    """
    if not pipeline.is_trained:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not ready. Retry in a few seconds.",
        )

    try:
        raw_readings = [r.model_dump() for r in request.readings]
        results = pipeline.predict(raw_readings, sensitivity=request.sensitivity)
    except Exception as e:
        logger.exception("Pipeline inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {str(e)}",
        )

    anomalies = [r for r in results if r["is_anomaly"]]
    anomaly_rate = round(len(anomalies) / len(results) * 100, 2)

    return BatchAnalysisResponse(
        total_readings=len(results),
        anomalies_detected=len(anomalies),
        anomaly_rate_pct=anomaly_rate,
        results=[AnomalyResult(**r) for r in results],
    )


@router.get(
    "/pipeline/status",
    tags=["Pipeline"],
    summary="Get pipeline metadata",
)
async def pipeline_status():
    """Return metadata about the loaded AI model."""
    return {
        "model_type": "IsolationForest",
        "feature_columns": [
            "speed_kmh", "engine_rpm", "engine_temp_c",
            "battery_voltage_v", "fuel_level_pct", "oil_pressure_bar",
        ],
        "pipeline_version": "1.0.0",
        "trained": pipeline.is_trained,
        "framework": "scikit-learn",
    }

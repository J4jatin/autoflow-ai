"""
AutoFlow AI — Anomaly Detection Pipeline
Uses Isolation Forest to detect anomalous vehicle sensor readings.
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "speed_kmh", "engine_rpm", "engine_temp_c",
    "battery_voltage_v", "fuel_level_pct", "oil_pressure_bar",
]

NORMAL_RANGES = {
    "speed_kmh":         (0, 200),
    "engine_rpm":        (600, 7000),
    "engine_temp_c":     (70, 110),
    "battery_voltage_v": (11.5, 14.8),
    "fuel_level_pct":    (5, 100),
    "oil_pressure_bar":  (1.5, 5.0),
}

MODEL_PATH = Path(__file__).parent / "model.joblib"
SCALER_PATH = Path(__file__).parent / "scaler.joblib"


def _generate_training_data(n_samples=2000):
    rng = np.random.default_rng(42)
    return np.column_stack([
        rng.normal(80, 30, n_samples).clip(0, 180),
        rng.normal(2500, 800, n_samples).clip(700, 6500),
        rng.normal(90, 8, n_samples).clip(72, 108),
        rng.normal(13.2, 0.5, n_samples).clip(12.0, 14.5),
        rng.uniform(20, 95, n_samples),
        rng.normal(3.0, 0.5, n_samples).clip(1.8, 4.8),
    ])


class AnomalyDetectionPipeline:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False

    def train(self, contamination=0.05):
        if MODEL_PATH.exists() and SCALER_PATH.exists():
            logger.info("Loading cached model from disk.")
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
        else:
            logger.info("Training Isolation Forest on synthetic data.")
            X = _generate_training_data()
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            self.model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42, n_jobs=-1)
            self.model.fit(X_scaled)
            joblib.dump(self.model, MODEL_PATH)
            joblib.dump(self.scaler, SCALER_PATH)
            logger.info("Model trained and saved.")
        self.is_trained = True

    def _flag_sensors(self, reading):
        flagged = []
        for sensor, (lo, hi) in NORMAL_RANGES.items():
            val = reading.get(sensor)
            if val is not None and not (lo <= val <= hi):
                flagged.append(sensor)
        return flagged

    def _score_to_severity(self, score, flagged_count):
        if flagged_count >= 5:
            return "critical"
        if flagged_count >= 3:
            return "high"
        if flagged_count == 2:
            return "medium"
        if score > -0.1 and flagged_count == 0:
            return "normal"
        if score > -0.25:
            return "low"
        if score > -0.4:
            return "medium"
        if score > -0.55:
            return "high"
        return "critical"

    def _normalise_score(self, raw_score):
        clipped = max(-0.8, min(0.2, raw_score))
        return round(1.0 - (clipped + 0.8) / 1.0, 4)

    def predict(self, readings, sensitivity=0.05):
        if not self.is_trained:
            raise RuntimeError("Pipeline not trained. Call .train() first.")
        X = np.array([[r[col] for col in FEATURE_COLUMNS] for r in readings])
        X_scaled = self.scaler.transform(X)
        raw_scores = self.model.decision_function(X_scaled)
        predictions = self.model.predict(X_scaled)
        results = []
        for i, reading in enumerate(readings):
            is_anomaly = bool(predictions[i] == -1)
            score = float(raw_scores[i])
            flagged = self._flag_sensors(reading)
            severity = self._score_to_severity(score, len(flagged))
            prob = self._normalise_score(score)
            results.append({
                "vehicle_id": reading.get("vehicle_id", "unknown"),
                "is_anomaly": is_anomaly,
                "anomaly_score": round(score, 6),
                "anomaly_probability": prob,
                "flagged_sensors": flagged,
                "severity": severity,
            })
        return results


pipeline = AnomalyDetectionPipeline()

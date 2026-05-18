"""
AutoFlow AI — Main Application Entry Point
FastAPI application for vehicle sensor anomaly detection.
Leverages Isolation Forest ML pipeline with AWS Lambda-compatible handler.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

from app.api.routes import router
from app.pipeline.detector import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Train the AI pipeline on startup, clean up on shutdown."""
    logger.info("AutoFlow AI starting — training anomaly detection pipeline...")
    pipeline.train()
    logger.info("Pipeline ready.")
    yield
    logger.info("AutoFlow AI shutting down.")


app = FastAPI(
    title="AutoFlow AI",
    description=(
        "Vehicle sensor anomaly detection API. "
        "Deploys an Isolation Forest ML pipeline via FastAPI on AWS Lambda + API Gateway. "
        "Built for AUDI AG internship application — Jattin Shah."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "AutoFlow AI",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


# AWS Lambda handler — Mangum wraps ASGI app for API Gateway
handler = Mangum(app, lifespan="on")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

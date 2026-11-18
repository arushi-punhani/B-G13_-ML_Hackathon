"""FastAPI application for PathSnap inference on reserved student-test trips."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.service import PathSnapService, UnknownTripError


logging.basicConfig(level=os.getenv("PATHSNAP_LOG_LEVEL", "INFO"))
logger = logging.getLogger("pathsnap.api")
ROOT = Path(__file__).resolve().parents[1]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = PathSnapService(root=ROOT)
    logger.info("PathSnap model loaded; reserved-trip API is ready.")
    yield


app = FastAPI(
    title="PathSnap Route Matching API",
    description=(
        "Runs the standalone geometry-based HMM/Viterbi decoder for a reserved "
        "student-test trip. Reference route data remains server-side."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

origins = [
    origin.strip()
    for origin in os.getenv(
        "PATHSNAP_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def get_service(request: Request) -> PathSnapService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Model service is not ready.")
    return service


@app.get("/health")
def health(request: Request) -> dict[str, Any]:
    service = getattr(request.app.state, "service", None)
    return {
        "status": "ok" if service is not None else "starting",
        "model_loaded": service is not None,
    }


@app.get("/trips")
def list_trips(request: Request) -> list[dict[str, Any]]:
    return get_service(request).list_trips()


@app.post("/predict/{trip_id}")
def predict_trip(trip_id: str, request: Request) -> dict[str, Any]:
    service = get_service(request)
    logger.info("Prediction requested for reserved trip %s", trip_id)
    try:
        result = service.predict_trip(trip_id)
    except UnknownTripError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'")) from exc
    except FileNotFoundError as exc:
        logger.exception("Trip data is incomplete for %s", trip_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except (ValueError, KeyError) as exc:
        logger.exception("Prediction failed for %s", trip_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    logger.info(
        "Prediction completed for %s: %s observations, route-membership proxy %.3f",
        trip_id,
        result["metrics"]["decoded_observations"],
        result["metrics"]["route_membership_proxy_rate"],
    )
    return result

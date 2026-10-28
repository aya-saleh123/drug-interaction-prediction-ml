
"""FastAPI backend for the drug-pair classification models."""

from __future__ import annotations

from functools import lru_cache
from typing import Any
import traceback

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference import DrugInteractionPredictor


app = FastAPI(
    title="Drug Pair Classification API",
    description=(
        "Predicts severity and 50 adverse-effect labels for two "
        "drug profiles from the project's dataset. "
        "For research and educational use only."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def predictor() -> DrugInteractionPredictor:
    """Load the models and artifacts once per process."""
    return DrugInteractionPredictor()


class PredictionRequest(BaseModel):
    drug_a_id: str = Field(
        min_length=1,
        description="A drug_id returned by GET /drugs",
    )
    drug_b_id: str = Field(
        min_length=1,
        description="A drug_id returned by GET /drugs",
    )


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Drug Pair Classification API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, str]:
    """Check whether the artifacts and models load successfully."""
    try:
        predictor()
        return {
            "status": "ok",
            "models": "loaded",
        }
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not load model artifacts: {exc}",
        ) from exc


@app.get("/drugs")
def list_drugs(
    search: str | None = Query(
        default=None,
        description="Search catalog IDs and available profile text",
    ),
    limit: int = Query(default=600, ge=1, le=600),
) -> list[dict[str, str]]:
    """Return selectable profiles from drug_catalog.csv."""
    try:
        return predictor().list_drugs(
            search=search,
            limit=limit,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load drug catalog: {exc}",
        ) from exc


@app.post("/predict")
def predict(
    request: PredictionRequest,
) -> dict[str, Any]:
    """Predict severity and adverse-effect labels for a pair."""
    try:
        return predictor().predict_pair(
            request.drug_a_id,
            request.drug_b_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc

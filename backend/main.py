
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
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
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
class PredictByNameRequest(BaseModel):
    drug_a: str = Field(
        min_length=1,
        description="Name or ID of first drug (e.g. 'Aspirin' or 'drug_0385')",
    )
    drug_b: str = Field(
        min_length=1,
        description="Name or ID of second drug (e.g. 'Warfarin' or 'drug_0209')",
    )


def _resolve_drug(pred: DrugInteractionPredictor, query: str) -> tuple[str, str]:
    """Resolve a drug by ID or Name using drug_lookup and drug_catalog."""
    import re
    q = query.strip().casefold()
    lookup = pred.drug_lookup

    # 1. Exact drug_id match
    m_id = lookup[lookup["drug_id"].str.casefold() == q]
    if not m_id.empty:
        return str(m_id.iloc[0]["drug_id"]), str(m_id.iloc[0]["drug_name"])

    # 2. Exact drug_name match
    m_name = lookup[lookup["drug_name"].str.casefold() == q]
    if not m_name.empty:
        return str(m_name.iloc[0]["drug_id"]), str(m_name.iloc[0]["drug_name"])

    # 3. Starts with drug_name
    m_starts = lookup[lookup["drug_name"].str.casefold().str.startswith(q)]
    if not m_starts.empty:
        return str(m_starts.iloc[0]["drug_id"]), str(m_starts.iloc[0]["drug_name"])

    # 4. Contains drug_name
    m_contains = lookup[lookup["drug_name"].str.casefold().str.contains(re.escape(q), na=False)]
    if not m_contains.empty:
        return str(m_contains.iloc[0]["drug_id"]), str(m_contains.iloc[0]["drug_name"])

    # 5. Fallback: Search in catalog
    cat_results = pred.list_drugs(search=query, limit=1)
    if cat_results:
        return cat_results[0]["drug_id"], cat_results[0].get("drug_name") or cat_results[0]["label"]

    raise ValueError(f"Could not find any drug matching '{query}'. Use GET /drugs?search={query} to explore available drugs.")


@app.post("/predict-by-name")
def predict_by_name(
    request: PredictByNameRequest,
) -> dict[str, Any]:
    """Predict interaction directly using drug names (ideal for Chatbots & n8n)."""
    try:
        p = predictor()
        drug_a_id, drug_a_name = _resolve_drug(p, request.drug_a)
        drug_b_id, drug_b_name = _resolve_drug(p, request.drug_b)

        result = p.predict_pair(drug_a_id, drug_b_id)
        result["resolved_inputs"] = {
            "drug_a": {"query": request.drug_a, "matched_name": drug_a_name, "id": drug_a_id},
            "drug_b": {"query": request.drug_b, "matched_name": drug_b_name, "id": drug_b_id},
        }
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

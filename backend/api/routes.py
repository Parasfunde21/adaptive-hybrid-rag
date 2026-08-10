import time

from fastapi import APIRouter
from fastapi import HTTPException

from api.schemas import (
    QueryRequest,
    QueryResponse
)

from services.analytics_service import analytics_service
from services.rag_pipeline import rag_pipeline


# ============================================================
# Router
# ============================================================

router = APIRouter()


# ============================================================
# ASK
# ============================================================

@router.post(
    "/ask",
    response_model=QueryResponse
)
def ask_question(
    request: QueryRequest
):

    start = time.perf_counter()

    try:

        # ====================================================
        # Run RAG Pipeline
        # ====================================================

        result = rag_pipeline.answer(
            query=request.query
        )


        # ====================================================
        # Processing Time
        # ====================================================

        elapsed = round(
            time.perf_counter() - start,
            3
        )


        # ====================================================
        # Build Response
        # ====================================================

        response = {
            "success": True,
            "processing_time": elapsed,
            **result
        }


        return response


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# ANALYTICS
# ============================================================

@router.get(
    "/analytics/summary"
)
def analytics_summary():

    return analytics_service.get_summary()


@router.get(
    "/analytics/weights"
)
def analytics_weights():

    return analytics_service.get_weight_distribution()


@router.get(
    "/analytics/models"
)
def analytics_models():

    return analytics_service.get_model_usage()


@router.get(
    "/analytics/recent"
)
def analytics_recent():

    return analytics_service.get_recent_queries(
        limit=20
    )
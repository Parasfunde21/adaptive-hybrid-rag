import time

from fastapi import APIRouter, HTTPException

from api.auth import router as auth_router
from api.user_routes import router as user_document_router
from api.user_chat import router as user_chat_router
from api.conversations import router as conversation_router

from api.schemas import (
    QueryRequest,
    QueryResponse,
)

from services.analytics_service import analytics_service
from services.rag_pipeline import rag_pipeline


# ============================================================
# Main Router
# ============================================================

router = APIRouter()


# ============================================================
# Authentication
# ============================================================

router.include_router(
    auth_router
)


# ============================================================
# User Documents
# ============================================================

router.include_router(
    user_document_router
)


# ============================================================
# User Chat
# ============================================================

router.include_router(
    user_chat_router
)


# ============================================================
# Conversations
# ============================================================

router.include_router(
    conversation_router
)


# ============================================================
# General ASK
# ============================================================

@router.post(
    "/ask",
    response_model=QueryResponse,
)
def ask_question(
    request: QueryRequest,
):

    start = time.perf_counter()

    try:

        result = rag_pipeline.answer(
            query=request.query
        )

        elapsed = round(
            time.perf_counter()
            - start,
            3,
        )

        return {
            "success": True,
            "processing_time": elapsed,
            **result,
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ============================================================
# Analytics
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
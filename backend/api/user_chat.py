import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.user_rag.user_pipeline import (
    user_rag_pipeline,
)

from services.user_rag.user_retriever import (
    user_adaptive_retriever,
)


router = APIRouter(
    prefix="/user",
    tags=["User RAG"],
)


# ============================================================
# Request
# ============================================================

class UserQuestionRequest(BaseModel):

    user_id: str = Field(
        min_length=1
    )

    query: str = Field(
        min_length=1
    )

    top_k: int = Field(
        default=10,
        ge=1,
        le=20
    )


# ============================================================
# List User Documents
# ============================================================

@router.get("/{user_id}/documents")
def get_user_documents(
    user_id: str
):

    try:

        collection = (
            user_adaptive_retriever.get_collection(
                user_id
            )
        )

        results = collection.get(
            include=[
                "metadatas",
            ]
        )

        metadatas = (
            results.get(
                "metadatas",
                []
            )
            or []
        )

        # ----------------------------------------------------
        # Group chunks by uploaded file
        # ----------------------------------------------------

        documents = {}

        for metadata in metadatas:

            metadata = metadata or {}

            file_id = (
                metadata.get("file_id")
                or metadata.get("document_id")
                or metadata.get("file_name")
            )

            file_name = (
                metadata.get("file_name")
                or metadata.get("document_id")
                or "Unknown document"
            )

            if not file_id:
                continue

            if file_id not in documents:

                documents[file_id] = {
                    "file_id":
                        file_id,

                    "file_name":
                        file_name,

                    "chunks":
                        0,
                }

            documents[file_id]["chunks"] += 1

        return {
            "success": True,
            "user_id": user_id,
            "documents": list(
                documents.values()
            ),
            "total_documents":
                len(documents),
            "total_chunks":
                len(metadatas),
        }

    except Exception as error:

        # ----------------------------------------------------
        # No collection = no uploaded documents.
        # Return an empty list instead of an error.
        # ----------------------------------------------------

        return {
            "success": True,
            "user_id": user_id,
            "documents": [],
            "total_documents": 0,
            "total_chunks": 0,
        }


# ============================================================
# Ask
# ============================================================

@router.post("/ask")
def ask_user_documents(
    request: UserQuestionRequest
):

    start = time.perf_counter()

    try:

        result = (
            user_rag_pipeline.answer(
                user_id=request.user_id,
                query=request.query,

                # IMPORTANT:
                # UserRAGPipeline.answer() uses
                # retrieval_candidate_k, not retrieval_top_k.
                retrieval_candidate_k=request.top_k,

                rerank_top_k=5,
            )
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        return {

            "success": True,

            **result,

            "processing_time":
                round(
                    elapsed,
                    3
                ),
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )
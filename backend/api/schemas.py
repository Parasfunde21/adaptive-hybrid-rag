from typing import Optional, List

from pydantic import BaseModel, Field


# ============================================================
# Query Request
# ============================================================

class QueryRequest(BaseModel):

    query: str


# ============================================================
# User RAG Question Request
# ============================================================

class UserQuestionRequest(BaseModel):

    user_id: str = Field(
        ...,
        min_length=1,
        description="Unique user identifier"
    )

    query: str = Field(
        ...,
        min_length=1,
        description="Question to answer from the user's uploaded documents"
    )


# ============================================================
# Adaptive Weights
# ============================================================

class WeightResponse(BaseModel):

    bm25: float

    dense: float


# ============================================================
# Citation
# ============================================================

class Citation(BaseModel):

    source: Optional[str] = None

    document_id: Optional[str] = None

    chunk_id: Optional[str] = None

    chunk_index: Optional[int] = None


# ============================================================
# Retrieved Document
# ============================================================

class RetrievedDocument(BaseModel):

    document: str

    rank: Optional[int] = None

    fusion_score: Optional[float] = None

    retrieved_by: List[str] = Field(
        default_factory=list
    )

    dense_similarity: Optional[float] = None

    bm25_score: Optional[float] = None

    dense_rank: Optional[int] = None

    bm25_rank: Optional[int] = None

    cross_score: Optional[float] = None

    cross_rank: Optional[int] = None

    source: Optional[str] = None

    document_id: Optional[str] = None

    chunk_id: Optional[str] = None

    chunk_index: Optional[int] = None

    citation: Optional[Citation] = None

    # User-RAG metadata

    file_id: Optional[str] = None

    file_name: Optional[str] = None

    document_type: Optional[str] = None


# ============================================================
# Latency
# ============================================================

class LatencyResponse(BaseModel):

    retrieval_ms: float = 0.0

    reranking_ms: float = 0.0

    prompt_ms: float = 0.0

    generation_ms: float = 0.0

    total_ms: float = 0.0


# ============================================================
# Query Response
# ============================================================

class QueryResponse(BaseModel):

    success: bool

    processing_time: float

    query: str

    answer: str

    weights: WeightResponse

    documents: List[RetrievedDocument]

    latency: Optional[LatencyResponse] = None

    model: Optional[str] = None

    class Config:

        extra = "allow"


# ============================================================
# User RAG Response
# ============================================================

class UserQuestionResponse(BaseModel):

    success: bool

    user_id: str

    query: str

    answer: str

    weights: WeightResponse

    documents: List[RetrievedDocument] = Field(
        default_factory=list
    )

    latency: Optional[LatencyResponse] = None

    model: Optional[str] = None

    processing_time: Optional[float] = None

    class Config:

        extra = "allow"
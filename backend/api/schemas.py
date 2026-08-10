from typing import Optional, List

from pydantic import BaseModel, Field


# ============================================================
# Query Request
# ============================================================

class QueryRequest(BaseModel):

    query: str


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


# ============================================================
# Latency
# ============================================================

class LatencyResponse(BaseModel):

    retrieval_ms: float

    reranking_ms: float

    prompt_ms: float

    generation_ms: float

    total_ms: float


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
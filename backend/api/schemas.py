from pydantic import BaseModel


class QueryRequest(BaseModel):

    query: str


class WeightResponse(BaseModel):

    bm25: float

    dense: float


class RetrievedDocument(BaseModel):

    document: str

    fusion_score: float

    cross_score: float

    retrieved_by: list[str]

    dense_similarity: float | None

    bm25_score: float | None

    dense_rank: int | None

    bm25_rank: int | None

    cross_rank: int


class QueryResponse(BaseModel):

    success: bool

    processing_time: float

    query: str

    answer: str

    weights: WeightResponse

    documents: list[RetrievedDocument]
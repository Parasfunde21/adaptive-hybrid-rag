import time

from fastapi import APIRouter
from fastapi import HTTPException

from api.schemas import (

    QueryRequest,

    QueryResponse

)

from services.rag_pipeline import rag_pipeline


router = APIRouter()


@router.post(

    "/ask",

    response_model=QueryResponse

)

def ask_question(

    request: QueryRequest

):

    start = time.perf_counter()

    try:

        result = rag_pipeline.answer(

            query=request.query

        )

        elapsed = round(

            time.perf_counter() - start,

            3

        )

        return {

            "success": True,

            "processing_time": elapsed,

            **result

        }

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )
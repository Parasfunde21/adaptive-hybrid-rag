from fastapi import FastAPI

from api.routes import router
from api.exceptions import global_exception_handler


app = FastAPI(

    title="Adaptive Hybrid RAG API",

    version="1.0.0",

    description="Query-Aware Adaptive Hybrid Retrieval-Augmented Generation"

)


app.include_router(router)


app.add_exception_handler(

    Exception,

    global_exception_handler

)


@app.get("/")

def home():

    return {

        "message": "Adaptive Hybrid RAG API is running."

    }


@app.get("/health")

def health():

    return {

        "status": "healthy"

    }


@app.get("/info")

def info():

    return {

        "project": "Adaptive Hybrid RAG",

        "llm": "qwen2.5:7b",

        "embedding_model": "all-MiniLM-L6-v2",

        "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",

        "fusion": "Adaptive Score Fusion",

        "version": "1.0.0"

    }
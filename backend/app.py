from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from api.exceptions import global_exception_handler


# ============================================================
# Application
# ============================================================

app = FastAPI(
    title="Adaptive Hybrid RAG API",
    version="1.0.0",
    description=(
        "Query-Aware Adaptive Hybrid "
        "Retrieval-Augmented Generation"
    ),
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],

    allow_credentials=True,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ],
)


# ============================================================
# API Routes
# ============================================================

app.include_router(
    router
)


# ============================================================
# Global Exception Handler
# ============================================================

app.add_exception_handler(
    Exception,
    global_exception_handler,
)


# ============================================================
# Home
# ============================================================

@app.get("/")
def home():

    return {
        "message":
            "Adaptive Hybrid RAG API is running."
    }


# ============================================================
# Health
# ============================================================

@app.get("/health")
def health():

    return {
        "status":
            "healthy"
    }


# ============================================================
# Project Information
# ============================================================

@app.get("/info")
def info():

    return {

        "project":
            "Adaptive Hybrid RAG",

        "llm":
            "qwen2.5:7b",

        "embedding_model":
            "all-MiniLM-L6-v2",

        "reranker":
            "cross-encoder/ms-marco-MiniLM-L-6-v2",

        "fusion":
            "Adaptive Score Fusion",

        "version":
            "1.0.0",
    }
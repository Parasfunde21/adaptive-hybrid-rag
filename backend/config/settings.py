from pathlib import Path


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BACKEND_DIR = PROJECT_ROOT / "backend"

DATA_DIR = PROJECT_ROOT / "data"

CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"

MODEL_DIR = PROJECT_ROOT / "backend" / "models"

TRAINING_DIR = PROJECT_ROOT / "backend" / "training"


# ============================================================
# ChromaDB
# ============================================================

# Multi-domain BEIR V2 collection
COLLECTION_NAME = "adaptive_rag_beir_v2"


# ============================================================
# Embedding Model
# ============================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ============================================================
# Retrieval
# ============================================================

# Final number of documents returned to the user
TOP_K = 10


# Number of candidates retrieved independently by
# BM25 and Dense retrieval before adaptive fusion.
#
# Research rationale:
# A larger candidate pool gives the adaptive fusion
# mechanism more opportunity to reorder documents.
#
# Retrieval:
#     BM25  -> 50 candidates
#     Dense -> 50 candidates
#              |
#              v
#       Adaptive Fusion
#              |
#              v
#          Top 10
#
CANDIDATE_K = 50
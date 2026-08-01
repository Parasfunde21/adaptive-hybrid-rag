from pathlib import Path

# ==============================
# Project Paths
# ==============================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BACKEND_DIR = PROJECT_ROOT / "backend"

DATA_DIR = PROJECT_ROOT / "data"

CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"

MODEL_DIR = PROJECT_ROOT / "backend" / "models"

TRAINING_DIR = PROJECT_ROOT / "backend" / "training"


# ==============================
# ChromaDB
# ==============================

COLLECTION_NAME = "documents"


# ==============================
# Embedding Model
# ==============================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ==============================
# Retrieval
# ==============================

TOP_K = 5
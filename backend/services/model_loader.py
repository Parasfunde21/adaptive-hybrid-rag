import os

from sentence_transformers import SentenceTransformer

from config.settings import (
    EMBEDDING_MODEL,
)


# ============================================================
# Local embedding model loader
# ============================================================

def load_embedding_model():
    """
    Load the embedding model strictly from local cache.

    This prevents repeated network calls to Hugging Face
    during FastAPI startup.
    """

    # Sentence Transformers / Transformers recognize this
    # environment variable for offline operation.
    os.environ.setdefault(
        "HF_HUB_OFFLINE",
        "1",
    )

    os.environ.setdefault(
        "TRANSFORMERS_OFFLINE",
        "1",
    )

    print(
        f"Loading local embedding model: "
        f"{EMBEDDING_MODEL}"
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        local_files_only=True,
    )

    print(
        "Local embedding model loaded successfully."
    )

    return model


embedding_model = load_embedding_model()
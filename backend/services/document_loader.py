import chromadb

from config.settings import (
    CHROMA_DB_DIR,
    COLLECTION_NAME
)


# ============================================================
# ChromaDB Client
# ============================================================

client = chromadb.PersistentClient(
    path=str(CHROMA_DB_DIR)
)


# ============================================================
# Configuration
# ============================================================

DEFAULT_BATCH_SIZE = 5000


# ============================================================
# Collection
# ============================================================

def get_collection():
    """
    Return the configured ChromaDB collection.
    """

    return client.get_collection(
        COLLECTION_NAME
    )


# ============================================================
# Load All Documents
# ============================================================

def get_all_documents(
    batch_size=DEFAULT_BATCH_SIZE
):
    """
    Load all indexed documents from ChromaDB
    safely in batches.

    This avoids ChromaDB/SQLite errors caused by
    attempting to retrieve a very large collection
    in a single query.

    Parameters
    ----------
    batch_size : int
        Number of documents retrieved per ChromaDB request.

    Returns
    -------
    list[str]
        All indexed documents.
    """

    collection = get_collection()

    total = collection.count()

    if total == 0:
        return []

    documents = []

    print(
        f"Loading {total} documents from ChromaDB "
        f"in batches of {batch_size}..."
    )

    for offset in range(
        0,
        total,
        batch_size
    ):

        limit = min(
            batch_size,
            total - offset
        )

        results = collection.get(
            include=["documents"],
            limit=limit,
            offset=offset
        )

        batch_documents = (
            results.get("documents")
            or []
        )

        documents.extend(
            batch_documents
        )

        loaded = len(documents)

        percentage = (
            loaded / total
        ) * 100

        print(
            f"Loaded {loaded:>6}/{total} "
            f"({percentage:6.2f}%)"
        )

    print(
        f"Successfully loaded "
        f"{len(documents)} documents."
    )

    return documents


# ============================================================
# Main Test
# ============================================================

if __name__ == "__main__":

    print(
        "Using ChromaDB:",
        CHROMA_DB_DIR
    )

    print(
        "Collection:",
        COLLECTION_NAME
    )

    docs = get_all_documents()

    print(
        f"\nTotal Documents: {len(docs)}"
    )

    if docs:

        print(
            "\nFirst Document:\n"
        )

        print(
            docs[0][:500]
        )
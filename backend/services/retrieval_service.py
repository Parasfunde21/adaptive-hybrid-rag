import chromadb

from sentence_transformers import SentenceTransformer

from config.settings import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
)


# ============================================================
# Embedding Model
# ============================================================

print(
    f"Loading embedding model: {EMBEDDING_MODEL}"
)

model = SentenceTransformer(
    EMBEDDING_MODEL
)


# ============================================================
# ChromaDB
# ============================================================

print(
    f"Opening ChromaDB collection: {COLLECTION_NAME}"
)

client = chromadb.PersistentClient(
    path=str(CHROMA_DB_DIR)
)


collection = client.get_collection(
    COLLECTION_NAME
)


# ============================================================
# Validation
# ============================================================

collection_count = collection.count()

print(
    f"Dense retrieval collection count: "
    f"{collection_count}"
)

if collection_count == 0:

    raise ValueError(
        f"ChromaDB collection '{COLLECTION_NAME}' "
        "is empty."
    )


# ============================================================
# Dense Search
# ============================================================

def dense_search(
    query: str,
    top_k: int = TOP_K
):

    """
    Perform dense semantic retrieval.

    Parameters
    ----------
    query : str
        User query.

    top_k : int
        Number of results to retrieve.

    Returns
    -------
    tuple
        documents,
        distances,
        ids,
        metadatas
    """

    if not query or not query.strip():

        return (
            [],
            [],
            [],
            []
        )


    # --------------------------------------------------------
    # Encode query
    # --------------------------------------------------------

    embedding = model.encode(
        query,
        normalize_embeddings=False
    ).tolist()


    # --------------------------------------------------------
    # Chroma similarity search
    # --------------------------------------------------------

    results = collection.query(

        query_embeddings=[
            embedding
        ],

        n_results=top_k,

        include=[
            "documents",
            "distances",
            "metadatas"
        ]
    )


    # --------------------------------------------------------
    # Extract results
    # --------------------------------------------------------

    documents = (
        results.get("documents", [[]])[0]
        or []
    )

    distances = (
        results.get("distances", [[]])[0]
        or []
    )

    ids = (
        results.get("ids", [[]])[0]
        or []
    )

    metadatas = (
        results.get("metadatas", [[]])[0]
        or []
    )


    return (
        documents,
        distances,
        ids,
        metadatas
    )


# ============================================================
# Manual Test
# ============================================================

if __name__ == "__main__":

    query = input(
        "Enter Query: "
    )


    docs, distances, ids, metadatas = dense_search(
        query,
        TOP_K
    )


    print()
    print("=" * 80)
    print("TOP DENSE RESULTS")
    print("=" * 80)


    for rank, (
        doc,
        distance,
        idx,
        metadata
    ) in enumerate(

        zip(
            docs,
            distances,
            ids,
            metadatas
        ),

        start=1
    ):

        print()
        print("-" * 80)

        print(
            f"Rank        : {rank}"
        )

        print(
            f"ID          : {idx}"
        )

        print(
            f"Distance    : {distance:.6f}"
        )

        print(
            f"Domain      : "
            f"{metadata.get('domain')}"
        )

        print(
            f"Dataset     : "
            f"{metadata.get('dataset')}"
        )

        print(
            f"Document ID : "
            f"{metadata.get('document_id')}"
        )

        print(
            f"Source      : "
            f"{metadata.get('source')}"
        )

        print()

        print(
            doc[:500]
        )
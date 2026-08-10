import chromadb

from sentence_transformers import SentenceTransformer

from config.settings import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL
)


# ============================================================
# Embedding Model
# ============================================================

model = SentenceTransformer(
    EMBEDDING_MODEL
)


# ============================================================
# ChromaDB
# ============================================================

client = chromadb.PersistentClient(
    path=str(CHROMA_DB_DIR)
)


collection = client.get_collection(
    COLLECTION_NAME
)


# ============================================================
# Dense Search
# ============================================================

def dense_search(
    query: str,
    top_k: int = 10
):

    """
    Dense semantic retrieval.

    Returns:

        documents
        distances
        ids
        metadatas
    """

    embedding = model.encode(
        query
    ).tolist()


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


    documents = results["documents"][0]

    distances = results["distances"][0]

    ids = results["ids"][0]

    metadatas = results["metadatas"][0]


    return (
        documents,
        distances,
        ids,
        metadatas
    )


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    query = input(
        "Enter Query: "
    )


    docs, distances, ids, metadatas = dense_search(
        query
    )


    print(
        "\nTop Dense Results\n"
    )


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

        print(
            "=" * 80
        )

        print(
            f"Rank       : {rank}"
        )

        print(
            f"ID         : {idx}"
        )

        print(
            f"Distance   : {distance:.6f}"
        )

        print(
            f"Source     : {metadata.get('source')}"
        )

        print(
            f"Chunk ID    : {metadata.get('chunk_id')}"
        )

        print(
            f"Chunk Index : {metadata.get('chunk_index')}"
        )

        print(
            "-" * 80
        )

        print(
            doc[:500]
        )

        print()
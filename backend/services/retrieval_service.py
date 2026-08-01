import chromadb

from sentence_transformers import SentenceTransformer

from config.settings import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL
)

model = SentenceTransformer(EMBEDDING_MODEL)

client = chromadb.PersistentClient(
    path=str(CHROMA_DB_DIR)
)

collection = client.get_collection(
    COLLECTION_NAME
)


def dense_search(query: str, top_k: int = 10):
    """
    Returns:
        documents,
        distances,
        ids
    """

    embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    )

    documents = results["documents"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    return documents, distances, ids


if __name__ == "__main__":

    query = input("Enter Query: ")

    docs, distances, ids = dense_search(query)

    print("\nTop Dense Results\n")

    for rank, (doc, distance, idx) in enumerate(
        zip(docs, distances, ids),
        start=1
    ):

        print("=" * 80)
        print(f"Rank      : {rank}")
        print(f"ID        : {idx}")
        print(f"Distance  : {distance:.6f}")
        print("-" * 80)
        print(doc[:500])
        print()
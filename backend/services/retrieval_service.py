import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model only once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="../../chroma_db")
collection = client.get_collection("documents")


def dense_search(query: str, top_k: int = 5):
    """
    Perform semantic search using ChromaDB.

    Args:
        query (str): User query
        top_k (int): Number of results

    Returns:
        list: Retrieved documents
        list: Similarity distances
    """

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    documents = results["documents"][0]
    distances = results["distances"][0]

    return documents, distances


if __name__ == "__main__":

    docs, scores = dense_search("What is Agile Scrum?")

    print("\nTop Results\n")

    for i, doc in enumerate(docs):

        print(f"\nResult {i+1}")
        print("-" * 60)
        print(doc[:300])
        print(f"\nDistance: {scores[i]}")
import chromadb

from config.settings import (
    CHROMA_DB_DIR,
    COLLECTION_NAME
)

client = chromadb.PersistentClient(
    path=str(CHROMA_DB_DIR)
)


def get_all_documents():
    """
    Load all indexed documents from ChromaDB.
    """

    collection = client.get_collection(
        COLLECTION_NAME
    )

    results = collection.get()

    return results["documents"]


if __name__ == "__main__":

    print("Using ChromaDB:", CHROMA_DB_DIR)

    docs = get_all_documents()

    print(f"\nTotal Documents: {len(docs)}")

    print("\nFirst Document:\n")

    print(docs[0][:500])
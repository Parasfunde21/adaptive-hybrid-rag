import chromadb

client = chromadb.PersistentClient(path="../../chroma_db")


def get_all_documents():
    """
    Load all indexed documents from ChromaDB.
    """

    collection = client.get_collection("documents")

    results = collection.get()

    return results["documents"]


if __name__ == "__main__":

    documents = get_all_documents()

    print(f"Total Documents: {len(documents)}")

    print("\nFirst Document:\n")

    print(documents[0][:300])
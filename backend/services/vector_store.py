import chromadb

from pdf_service import extract_text, clean_text
from chunk_service import chunk_text
from embedding_service import generate_embeddings


# Persistent ChromaDB Client
client = chromadb.PersistentClient(path="../../chroma_db")


def store_embeddings(chunks, embeddings, source="document.pdf"):
    """
    Store chunks and embeddings into ChromaDB.
    """

    try:
        client.delete_collection("documents")
    except:
        pass

    collection = client.get_or_create_collection("documents")

    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[str(i)],
            documents=[chunk],
            embeddings=[embeddings[i].tolist()],
            metadatas=[{"source": source}]
        )

    return collection.count()


def load_collection():
    """
    Return ChromaDB collection.
    """

    return client.get_collection("documents")


def index_document(pdf_path):
    """
    Complete indexing pipeline.

    PDF
        ↓
    Extract
        ↓
    Clean
        ↓
    Chunk
        ↓
    Embed
        ↓
    Store
    """

    print("Extracting text...")

    text = extract_text(pdf_path)

    print("Cleaning text...")

    cleaned = clean_text(text)

    print("Creating chunks...")

    chunks = chunk_text(cleaned)

    print("Generating embeddings...")

    embeddings = generate_embeddings(chunks)

    print("Storing into ChromaDB...")

    total = store_embeddings(
        chunks,
        embeddings,
        source=pdf_path.split("/")[-1]
    )

    print(f"\nSuccessfully indexed {total} chunks.")

    return total


if __name__ == "__main__":

    pdf = "../../data/raw/sample.pdf"

    index_document(pdf)
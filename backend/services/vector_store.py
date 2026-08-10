import chromadb

from services.pdf_service import (
    extract_text,
    clean_text
)

from services.chunk_service import chunk_text

from services.embedding_service import (
    generate_embeddings
)

from config.settings import (
    CHROMA_DB_DIR,
    COLLECTION_NAME
)


# ============================================================
# Persistent ChromaDB Client
# ============================================================

client = chromadb.PersistentClient(
    path=str(CHROMA_DB_DIR)
)


# ============================================================
# Store Embeddings
# ============================================================

def store_embeddings(
    chunks,
    embeddings,
    source="document.pdf"
):
    """
    Store document chunks, embeddings and citation metadata.

    Metadata stored for every chunk:

        source
        document_id
        chunk_id
        chunk_index
    """

    # Remove the old collection so the metadata schema is
    # rebuilt cleanly.
    try:
        client.delete_collection(
            COLLECTION_NAME
        )

        print(
            f"Deleted old collection: {COLLECTION_NAME}"
        )

    except Exception:
        pass


    collection = client.get_or_create_collection(
        COLLECTION_NAME
    )


    for i, chunk in enumerate(chunks):

        chunk_id = f"{source}_chunk_{i}"


        collection.add(

            ids=[
                chunk_id
            ],

            documents=[
                chunk
            ],

            embeddings=[
                embeddings[i].tolist()
            ],

            metadatas=[

                {
                    "source": source,

                    "document_id": source,

                    "chunk_id": chunk_id,

                    "chunk_index": i
                }

            ]
        )


    return collection.count()


# ============================================================
# Load Collection
# ============================================================

def load_collection():

    return client.get_collection(
        COLLECTION_NAME
    )


# ============================================================
# Index Document
# ============================================================

def index_document(
    pdf_path
):

    """
    Complete PDF indexing pipeline.

    PDF
      ↓
    Extract text
      ↓
    Clean text
      ↓
    Chunk
      ↓
    Generate embeddings
      ↓
    Store in ChromaDB
    """

    print(
        "\nChromaDB path:"
    )

    print(
        CHROMA_DB_DIR
    )


    print(
        "\nCollection:"
    )

    print(
        COLLECTION_NAME
    )


    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    print(
        "\nExtracting text..."
    )

    text = extract_text(
        pdf_path
    )


    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    print(
        "Cleaning text..."
    )

    cleaned = clean_text(
        text
    )


    # --------------------------------------------------------
    # Chunk
    # --------------------------------------------------------

    print(
        "Creating chunks..."
    )

    chunks = chunk_text(
        cleaned
    )


    print(
        f"Created {len(chunks)} chunks."
    )


    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    print(
        "Generating embeddings..."
    )

    embeddings = generate_embeddings(
        chunks
    )


    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    print(
        "Storing into ChromaDB..."
    )


    source = (
        str(pdf_path)
        .replace("\\", "/")
        .split("/")[-1]
    )


    total = store_embeddings(

        chunks,

        embeddings,

        source=source
    )


    print(
        f"\nSuccessfully indexed {total} chunks."
    )


# ============================================================
# Test / Manual Indexing
# ============================================================

if __name__ == "__main__":

    pdf = "../data/raw/sample.pdf"

    index_document(
        pdf
    )
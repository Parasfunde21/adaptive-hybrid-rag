from pathlib import Path
import json
import time

import chromadb
from sentence_transformers import SentenceTransformer

from config.settings import (
    CHROMA_DB_DIR,
    EMBEDDING_MODEL,
)


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    BASE_DIR
    / "evaluation"
    / "dataset_v2"
    / "processed"
)


# ============================================================
# Dedicated V2 Collection
#
# IMPORTANT:
# Do NOT overwrite the existing collection.
# ============================================================

COLLECTION_NAME_V2 = "adaptive_rag_beir_v2"


# ============================================================
# Datasets
# ============================================================

DATASETS = [
    "scifact",
    "fiqa",
    "nfcorpus",
    "arguana",
]


# ============================================================
# Embedding configuration
# ============================================================

BATCH_SIZE = 64


# ============================================================
# Helpers
# ============================================================

def load_jsonl(path):
    """
    Load a JSONL file into a list of dictionaries.
    """

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            rows.append(
                json.loads(line)
            )

    return rows


# ============================================================
# Load all BEIR documents
# ============================================================

def load_documents():
    """
    Load all documents from the four BEIR datasets.

    Each original BEIR document remains one retrieval unit.
    No chunking is performed because BEIR qrels reference
    the original document IDs.
    """

    documents = []

    print()
    print("=" * 80)
    print("LOADING MULTI-DOMAIN BEIR CORPUS")
    print("=" * 80)

    for dataset_name in DATASETS:

        corpus_path = (
            PROCESSED_DIR
            / dataset_name
            / "corpus.jsonl"
        )

        if not corpus_path.exists():

            raise FileNotFoundError(
                f"Corpus not found:\n{corpus_path}"
            )

        rows = load_jsonl(
            corpus_path
        )

        print(
            f"{dataset_name:12} : "
            f"{len(rows):>8} documents"
        )

        for row in rows:

            document_id = str(
                row["document_id"]
            )

            title = (
                row.get("title")
                or ""
            )

            text = (
                row.get("text")
                or ""
            )

            dataset = (
                row.get("dataset")
                or dataset_name
            )

            domain = (
                row.get("domain")
                or "unknown"
            )

            source = (
                row.get("source")
                or "BEIR"
            )

            # ------------------------------------------------
            # Keep title + text together for semantic retrieval
            # ------------------------------------------------

            if title.strip():

                document_text = (
                    title.strip()
                    + "\n\n"
                    + text.strip()
                )

            else:

                document_text = (
                    text.strip()
                )

            if not document_text:

                continue

            documents.append(
                {
                    "document_id":
                        document_id,

                    "text":
                        document_text,

                    "title":
                        title,

                    "dataset":
                        dataset,

                    "domain":
                        domain,

                    "source":
                        source,
                }
            )

    print()
    print(
        f"Total documents loaded: "
        f"{len(documents)}"
    )

    return documents


# ============================================================
# Validate document IDs
# ============================================================

def validate_documents(documents):
    """
    Verify that document IDs are unique within the
    combined corpus.

    Dataset name is included in the internal Chroma ID
    so identical IDs from different datasets cannot collide.
    """

    seen = set()

    duplicate_count = 0

    for document in documents:

        key = (
            document["dataset"],
            document["document_id"]
        )

        if key in seen:

            duplicate_count += 1

        seen.add(key)

    print()
    print("=" * 80)
    print("DOCUMENT VALIDATION")
    print("=" * 80)

    print(
        f"Unique dataset/document pairs : "
        f"{len(seen)}"
    )

    print(
        f"Duplicate pairs               : "
        f"{duplicate_count}"
    )

    if duplicate_count > 0:

        raise ValueError(
            "Duplicate dataset/document IDs detected."
        )


# ============================================================
# Create ChromaDB collection
# ============================================================

def create_collection(client):
    """
    Create a dedicated V2 ChromaDB collection.

    Existing collections are NOT modified.
    """

    existing = [
        collection.name
        for collection
        in client.list_collections()
    ]

    if COLLECTION_NAME_V2 in existing:

        print()
        print(
            f"Collection already exists: "
            f"{COLLECTION_NAME_V2}"
        )

        answer = input(
            "Delete and rebuild it? [y/N]: "
        ).strip().lower()

        if answer == "y":

            client.delete_collection(
                COLLECTION_NAME_V2
            )

            print(
                "Existing V2 collection deleted."
            )

        else:

            raise RuntimeError(
                "V2 collection already exists. "
                "Rebuild cancelled."
            )

    collection = client.create_collection(
        name=COLLECTION_NAME_V2,
        metadata={
            "description":
                "Multi-domain BEIR corpus "
                "for Adaptive Hybrid RAG evaluation",

            "datasets":
                ",".join(DATASETS),

            "document_count":
                len(DATASETS),
        }
    )

    return collection


# ============================================================
# Index corpus
# ============================================================

def index_documents(
    collection,
    model,
    documents
):
    """
    Generate embeddings and insert documents into ChromaDB.

    Uses batches to avoid excessive RAM usage.
    """

    total = len(documents)

    print()
    print("=" * 80)
    print("GENERATING EMBEDDINGS AND INDEXING")
    print("=" * 80)

    print(
        f"Embedding model : {EMBEDDING_MODEL}"
    )

    print(
        f"Batch size      : {BATCH_SIZE}"
    )

    print(
        f"Documents       : {total}"
    )

    start_time = time.perf_counter()

    for start in range(
        0,
        total,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            total
        )

        batch = documents[
            start:end
        ]

        texts = [
            item["text"]
            for item in batch
        ]

        # ----------------------------------------------------
        # Generate embeddings
        # ----------------------------------------------------

        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        ids = []

        metadatas = []

        for item in batch:

            # ------------------------------------------------
            # Dataset-qualified Chroma ID
            #
            # Original BEIR document ID is preserved in
            # metadata for evaluation.
            # ------------------------------------------------

            chroma_id = (
                f"{item['dataset']}:"
                f"{item['document_id']}"
            )

            ids.append(
                chroma_id
            )

            metadatas.append(
                {
                    "document_id":
                        item["document_id"],

                    "dataset":
                        item["dataset"],

                    "domain":
                        item["domain"],

                    "source":
                        item["source"],

                    "title":
                        item["title"],
                }
            )

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

        processed = end

        elapsed = (
            time.perf_counter()
            - start_time
        )

        rate = (
            processed / elapsed
            if elapsed > 0
            else 0
        )

        print(
            f"Indexed "
            f"{processed:>6}/{total} "
            f"({processed / total * 100:6.2f}%) "
            f"| {rate:7.1f} docs/sec"
        )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print()
    print(
        f"Indexing completed in "
        f"{elapsed:.2f} seconds."
    )


# ============================================================
# Dataset statistics
# ============================================================

def print_statistics(
    collection,
    documents
):
    """
    Print final index statistics by domain.
    """

    print()
    print("=" * 80)
    print("INDEX STATISTICS")
    print("=" * 80)

    counts = {}

    for document in documents:

        domain = document["domain"]

        counts[domain] = (
            counts.get(domain, 0)
            + 1
        )

    for domain, count in sorted(
        counts.items()
    ):

        print(
            f"{domain:20} : "
            f"{count:>8} documents"
        )

    print()
    print(
        f"Chroma collection : "
        f"{COLLECTION_NAME_V2}"
    )

    print(
        f"Chroma count      : "
        f"{collection.count()}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 80)
    print("ADAPTIVE HYBRID RAG")
    print("MULTI-DOMAIN BEIR V2 INDEX BUILDER")
    print("=" * 80)

    print()
    print(
        f"Processed directory:\n"
        f"{PROCESSED_DIR}"
    )

    print()
    print(
        f"ChromaDB directory:\n"
        f"{CHROMA_DB_DIR}"
    )

    print()
    print(
        f"Collection:\n"
        f"{COLLECTION_NAME_V2}"
    )

    # --------------------------------------------------------
    # Load documents
    # --------------------------------------------------------

    documents = load_documents()

    if not documents:

        raise RuntimeError(
            "No documents were loaded."
        )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_documents(
        documents
    )

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("LOADING EMBEDDING MODEL")
    print("=" * 80)

    print(
        f"Model: {EMBEDDING_MODEL}"
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print(
        "Embedding model loaded."
    )

    # --------------------------------------------------------
    # ChromaDB
    # --------------------------------------------------------

    client = chromadb.PersistentClient(
        path=str(
            CHROMA_DB_DIR
        )
    )

    collection = create_collection(
        client
    )

    # --------------------------------------------------------
    # Index
    # --------------------------------------------------------

    index_documents(
        collection,
        model,
        documents
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print_statistics(
        collection,
        documents
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    expected = len(
        documents
    )

    actual = collection.count()

    print()
    print("=" * 80)
    print("FINAL VALIDATION")
    print("=" * 80)

    print(
        f"Expected documents : {expected}"
    )

    print(
        f"Indexed documents  : {actual}"
    )

    if expected != actual:

        raise RuntimeError(
            "Index validation failed: "
            "document counts do not match."
        )

    print()
    print(
        "MULTI-DOMAIN BEIR V2 INDEX "
        "CREATED SUCCESSFULLY."
    )

    print()
    print(
        f"Collection: "
        f"{COLLECTION_NAME_V2}"
    )


if __name__ == "__main__":
    main()
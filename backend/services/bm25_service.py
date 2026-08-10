from rank_bm25 import BM25Okapi
import numpy as np
import chromadb

from config.settings import (
    CHROMA_DB_DIR,
    COLLECTION_NAME
)


# ============================================================
# BM25 Retriever
# ============================================================

class BM25Retriever:

    def __init__(self):

        """
        Load indexed ChromaDB chunks and build BM25 index.

        Each BM25 document keeps its original ChromaDB:
            - document
            - id
            - metadata

        This prevents citation information from being lost
        during lexical retrieval.
        """

        # ----------------------------------------------------
        # ChromaDB
        # ----------------------------------------------------

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DB_DIR)
        )

        self.collection = self.client.get_collection(
            COLLECTION_NAME
        )

        # ----------------------------------------------------
        # Load complete records
        # ----------------------------------------------------

        results = self.collection.get(
            include=[
                "documents",
                "metadatas"
            ]
        )

        self.documents = (
            results.get("documents", [])
            or []
        )

        self.ids = (
            results.get("ids", [])
            or []
        )

        self.metadatas = (
            results.get("metadatas", [])
            or []
        )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if not self.documents:

            raise ValueError(
                "No documents found in ChromaDB. "
                "Run the vector store indexing first."
            )

        # ----------------------------------------------------
        # BM25 tokenization
        # ----------------------------------------------------

        self.tokenized_docs = [

            doc.lower().split()

            for doc in self.documents

        ]

        # ----------------------------------------------------
        # Build BM25
        # ----------------------------------------------------

        self.bm25 = BM25Okapi(
            self.tokenized_docs
        )


    # ========================================================
    # Search
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = 10
    ):

        """
        Returns:

            documents
            scores
            ids
            metadatas
        """

        # ----------------------------------------------------
        # Tokenize query
        # ----------------------------------------------------

        tokenized_query = (
            query.lower().split()
        )

        # ----------------------------------------------------
        # Calculate BM25 scores
        # ----------------------------------------------------

        scores = self.bm25.get_scores(
            tokenized_query
        )

        # ----------------------------------------------------
        # Top results
        # ----------------------------------------------------

        top_indices = (

            np.argsort(scores)[::-1]

            [:top_k]

        )

        # ----------------------------------------------------
        # Build result lists
        # ----------------------------------------------------

        documents = [

            self.documents[i]

            for i in top_indices

        ]

        top_scores = [

            float(scores[i])

            for i in top_indices

        ]

        ids = [

            self.ids[i]

            for i in top_indices

        ]

        metadatas = [

            self.metadatas[i]

            for i in top_indices

        ]

        return (

            documents,

            top_scores,

            ids,

            metadatas

        )


# ============================================================
# Global Retriever
# ============================================================

bm25_retriever = BM25Retriever()


# ============================================================
# Public Search Function
# ============================================================

def bm25_search(
    query: str,
    top_k: int = 10
):

    return bm25_retriever.search(
        query,
        top_k
    )


# ============================================================
# Manual Test
# ============================================================

if __name__ == "__main__":

    query = input(
        "Enter your query: "
    )

    (
        docs,
        scores,
        ids,
        metadatas
    ) = bm25_search(
        query
    )

    print(
        "\nTop BM25 Results\n"
    )

    for rank, (
        doc,
        score,
        idx,
        metadata
    ) in enumerate(

        zip(
            docs,
            scores,
            ids,
            metadatas
        ),

        start=1

    ):

        print(
            "=" * 80
        )

        print(
            f"Rank        : {rank}"
        )

        print(
            f"ID          : {idx}"
        )

        print(
            f"Score       : {score:.4f}"
        )

        print(
            f"Source      : "
            f"{metadata.get('source')}"
        )

        print(
            f"Document ID : "
            f"{metadata.get('document_id')}"
        )

        print(
            f"Chunk ID    : "
            f"{metadata.get('chunk_id')}"
        )

        print(
            f"Chunk Index : "
            f"{metadata.get('chunk_index')}"
        )

        print(
            "-" * 80
        )

        print(
            doc[:500]
        )

        print()
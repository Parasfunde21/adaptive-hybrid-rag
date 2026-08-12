import os
import pickle
import hashlib
from pathlib import Path

import chromadb
import numpy as np

from rank_bm25 import BM25Okapi

from config.settings import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    TOP_K,
)


# ============================================================
# Persistent BM25 Retriever
# ============================================================

class BM25Retriever:

    CACHE_VERSION = "bm25_v2"

    def __init__(
        self,
        batch_size=5000,
        cache_dir=None
    ):

        """
        BM25 retriever over the complete ChromaDB collection.

        The BM25 index is persisted to disk so that the 75K+
        document collection does not need to be tokenized and
        indexed every time the application starts.

        ChromaDB is still loaded in batches when creating the
        cache because SQLite/ChromaDB can fail when attempting
        to retrieve a very large collection in one operation.
        """

        self.batch_size = batch_size

        # ----------------------------------------------------
        # Cache directory
        # ----------------------------------------------------

        if cache_dir is None:

            cache_dir = (
                Path(CHROMA_DB_DIR).resolve().parent
                / "bm25_cache"
            )

        self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        safe_collection_name = (
            str(COLLECTION_NAME)
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )

        self.cache_path = (
            self.cache_dir
            / f"{safe_collection_name}.pkl"
        )

        # ----------------------------------------------------
        # ChromaDB
        # ----------------------------------------------------

        print(
            f"Opening BM25 collection: "
            f"{COLLECTION_NAME}"
        )

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DB_DIR)
        )

        self.collection = self.client.get_collection(
            COLLECTION_NAME
        )

        # ----------------------------------------------------
        # Collection validation
        # ----------------------------------------------------

        collection_count = (
            self.collection.count()
        )

        print(
            f"BM25 collection count: "
            f"{collection_count}"
        )

        if collection_count == 0:

            raise ValueError(
                f"ChromaDB collection "
                f"'{COLLECTION_NAME}' is empty."
            )

        self.collection_count = collection_count

        # ----------------------------------------------------
        # Try persistent cache
        # ----------------------------------------------------

        if self._load_cache():

            print()
            print(
                "BM25 persistent cache loaded successfully."
            )

            print(
                f"Cache: {self.cache_path}"
            )

            print(
                f"Documents in cache: "
                f"{len(self.documents)}"
            )

            print(
                "BM25 index ready."
            )

            return

        # ----------------------------------------------------
        # Cache unavailable / invalid
        # ----------------------------------------------------

        print()

        print(
            "No valid BM25 cache found."
        )

        print(
            "Building BM25 index from ChromaDB..."
        )

        self._load_documents()

        self._build_index()

        self._save_cache()

        print()
        print(
            "BM25 index ready."
        )

    # ========================================================
    # Collection fingerprint
    # ========================================================

    def _collection_fingerprint(self):

        """
        Create a lightweight fingerprint for the current
        ChromaDB collection.

        We intentionally avoid collection.get() without a
        limit because the collection contains 75K+ documents.

        Three small reads are enough to detect normal dataset
        replacement/change scenarios.
        """

        count = self.collection.count()

        if count == 0:

            return "empty"

        positions = sorted(
            set(
                [
                    0,
                    count // 2,
                    count - 1
                ]
            )
        )

        sampled_ids = []

        for offset in positions:

            result = self.collection.get(
                limit=1,
                offset=offset,
                include=[]
            )

            ids = (
                result.get(
                    "ids",
                    []
                )
                or []
            )

            if ids:

                sampled_ids.append(
                    str(ids[0])
                )

        payload = (
            f"{COLLECTION_NAME}|"
            f"{count}|"
            f"{'|'.join(sampled_ids)}"
        )

        return hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()

    # ========================================================
    # Cache loading
    # ========================================================

    def _load_cache(self):

        """
        Load the persisted BM25 object if the cache matches
        the current ChromaDB collection.
        """

        if not self.cache_path.exists():

            return False

        print()
        print(
            f"Checking BM25 cache:"
        )

        print(
            f"{self.cache_path}"
        )

        try:

            with open(
                self.cache_path,
                "rb"
            ) as file:

                cache = pickle.load(file)

        except Exception as error:

            print(
                "[WARNING] Could not load BM25 cache."
            )

            print(
                f"Reason: {error}"
            )

            return False

        # ----------------------------------------------------
        # Validate cache structure
        # ----------------------------------------------------

        if not isinstance(
            cache,
            dict
        ):

            print(
                "[WARNING] Invalid BM25 cache format."
            )

            return False

        if cache.get(
            "cache_version"
        ) != self.CACHE_VERSION:

            print(
                "[WARNING] BM25 cache version mismatch."
            )

            return False

        if cache.get(
            "collection_name"
        ) != COLLECTION_NAME:

            print(
                "[WARNING] BM25 cache belongs to another "
                "ChromaDB collection."
            )

            return False

        if cache.get(
            "collection_count"
        ) != self.collection_count:

            print(
                "[WARNING] BM25 cache document count "
                "does not match the current collection."
            )

            return False

        # ----------------------------------------------------
        # Validate collection fingerprint
        # ----------------------------------------------------

        try:

            current_fingerprint = (
                self._collection_fingerprint()
            )

        except Exception as error:

            print(
                "[WARNING] Could not validate BM25 "
                "collection fingerprint."
            )

            print(
                f"Reason: {error}"
            )

            return False

        if cache.get(
            "collection_fingerprint"
        ) != current_fingerprint:

            print(
                "[WARNING] BM25 cache fingerprint "
                "does not match the current collection."
            )

            return False

        # ----------------------------------------------------
        # Restore data
        # ----------------------------------------------------

        documents = cache.get(
            "documents"
        )

        ids = cache.get(
            "ids"
        )

        metadatas = cache.get(
            "metadatas"
        )

        tokenized_docs = cache.get(
            "tokenized_docs"
        )

        bm25 = cache.get(
            "bm25"
        )

        if (
            documents is None
            or ids is None
            or metadatas is None
            or tokenized_docs is None
            or bm25 is None
        ):

            print(
                "[WARNING] BM25 cache is missing "
                "required data."
            )

            return False

        # ----------------------------------------------------
        # Validate sizes
        # ----------------------------------------------------

        if not (
            len(documents)
            ==
            len(ids)
            ==
            len(metadatas)
            ==
            len(tokenized_docs)
            ==
            self.collection_count
        ):

            print(
                "[WARNING] BM25 cache contains inconsistent "
                "document counts."
            )

            return False

        self.documents = documents
        self.ids = ids
        self.metadatas = metadatas
        self.tokenized_docs = tokenized_docs
        self.bm25 = bm25

        return True

    # ========================================================
    # Load documents from ChromaDB
    # ========================================================

    def _load_documents(self):

        """
        Load all documents from ChromaDB in safe batches.
        """

        self.documents = []
        self.ids = []
        self.metadatas = []

        collection_count = (
            self.collection_count
        )

        print()

        print(
            f"Loading {collection_count} documents "
            f"in batches of {self.batch_size}..."
        )

        for offset in range(
            0,
            collection_count,
            self.batch_size
        ):

            end = min(
                offset + self.batch_size,
                collection_count
            )

            results = self.collection.get(

                limit=self.batch_size,

                offset=offset,

                include=[
                    "documents",
                    "metadatas"
                ]
            )

            batch_documents = (
                results.get(
                    "documents",
                    []
                )
                or []
            )

            batch_ids = (
                results.get(
                    "ids",
                    []
                )
                or []
            )

            batch_metadatas = (
                results.get(
                    "metadatas",
                    []
                )
                or []
            )

            self.documents.extend(
                batch_documents
            )

            self.ids.extend(
                batch_ids
            )

            self.metadatas.extend(
                batch_metadatas
            )

            percentage = (
                end / collection_count
            ) * 100

            print(
                f"Loaded "
                f"{end:>6}/{collection_count} "
                f"({percentage:6.2f}%)"
            )

        # ----------------------------------------------------
        # Validate loaded data
        # ----------------------------------------------------

        if not self.documents:

            raise ValueError(
                "No documents were loaded "
                "from ChromaDB."
            )

        if len(self.documents) != len(
            self.ids
        ):

            raise ValueError(
                "Document/ID count mismatch: "
                f"{len(self.documents)} documents vs "
                f"{len(self.ids)} IDs."
            )

        if len(self.documents) != len(
            self.metadatas
        ):

            raise ValueError(
                "Document/metadata count mismatch: "
                f"{len(self.documents)} documents vs "
                f"{len(self.metadatas)} metadata records."
            )

        if len(self.documents) != (
            self.collection_count
        ):

            raise ValueError(
                "Loaded document count does not match "
                "the ChromaDB collection count: "
                f"{len(self.documents)} vs "
                f"{self.collection_count}."
            )

        print()

        print(
            f"Successfully loaded "
            f"{len(self.documents)} documents."
        )

    # ========================================================
    # Build BM25
    # ========================================================

    def _build_index(self):

        """
        Tokenize the corpus and construct BM25Okapi.
        """

        print(
            "Tokenizing documents for BM25..."
        )

        self.tokenized_docs = [

            doc.lower().split()

            for doc in self.documents

        ]

        print(
            "Building BM25 index..."
        )

        self.bm25 = BM25Okapi(
            self.tokenized_docs
        )

    # ========================================================
    # Save cache
    # ========================================================

    def _save_cache(self):

        """
        Persist the complete BM25 index and supporting data.

        A temporary file is written first and then atomically
        replaced so an interrupted write does not normally leave
        a half-written cache.
        """

        print()

        print(
            "Saving persistent BM25 cache..."
        )

        try:

            fingerprint = (
                self._collection_fingerprint()
            )

            cache = {

                "cache_version":
                    self.CACHE_VERSION,

                "collection_name":
                    COLLECTION_NAME,

                "collection_count":
                    self.collection_count,

                "collection_fingerprint":
                    fingerprint,

                "documents":
                    self.documents,

                "ids":
                    self.ids,

                "metadatas":
                    self.metadatas,

                "tokenized_docs":
                    self.tokenized_docs,

                "bm25":
                    self.bm25
            }

            temporary_path = (
                self.cache_path.with_suffix(
                    ".tmp"
                )
            )

            with open(
                temporary_path,
                "wb"
            ) as file:

                pickle.dump(
                    cache,
                    file,
                    protocol=pickle.HIGHEST_PROTOCOL
                )

            os.replace(
                temporary_path,
                self.cache_path
            )

            cache_size_mb = (
                self.cache_path.stat().st_size
                /
                (
                    1024 * 1024
                )
            )

            print(
                "BM25 cache saved successfully."
            )

            print(
                f"Cache path : "
                f"{self.cache_path}"
            )

            print(
                f"Cache size : "
                f"{cache_size_mb:.2f} MB"
            )

        except Exception as error:

            print(
                "[WARNING] Failed to save BM25 cache."
            )

            print(
                f"Reason: {error}"
            )

            print(
                "The BM25 index will still work for "
                "the current process."
            )

    # ========================================================
    # Search
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = TOP_K
    ):

        """
        Perform BM25 lexical retrieval.

        Returns
        -------
        tuple
            documents,
            scores,
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
        # Top-K indices
        # ----------------------------------------------------

        top_indices = (

            np.argsort(scores)[::-1]

            [:top_k]

        )

        # ----------------------------------------------------
        # Build results
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

print()

print(
    "=" * 80
)

print(
    "INITIALIZING BM25 RETRIEVER"
)

print(
    "=" * 80
)

bm25_retriever = BM25Retriever(
    batch_size=5000
)


# ============================================================
# Public Search Function
# ============================================================

def bm25_search(
    query: str,
    top_k: int = TOP_K
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
        query,
        TOP_K
    )

    print()

    print(
        "=" * 80
    )

    print(
        "TOP BM25 RESULTS"
    )

    print(
        "=" * 80
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

        print()

        print(
            "-" * 80
        )

        print(
            f"Rank        : {rank}"
        )

        print(
            f"ID          : {idx}"
        )

        print(
            f"Score       : {score:.6f}"
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
            f"Source      : "
            f"{metadata.get('source')}"
        )

        print(
            f"Document ID : "
            f"{metadata.get('document_id')}"
        )

        print()

        print(
            doc[:500]
        )
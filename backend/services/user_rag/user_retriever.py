import re
import time

import chromadb
import numpy as np

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from config.settings import (
    CHROMA_DB_DIR,
    EMBEDDING_MODEL,
    TOP_K,
    CANDIDATE_K,
)

from services.adaptive_predictor_v4 import (
    adaptive_predictor_v4,
)

from services.retrieval_feature_service import (
    retrieval_feature_service,
)

from services.fusion_service import (
    fusion_service,
)


# ============================================================
# User Adaptive Retriever
# ============================================================

class UserAdaptiveRetriever:

    """
    Adaptive retriever for user-uploaded documents.

    Pipeline:

        User Query
            |
            +------------------+
            |                  |
            v                  v
          BM25              Dense
            |                  |
            +--------+---------+
                     |
                     v
            Retrieval Features
                     |
                     v
             Adaptive Predictor V4
                     |
                +----+----+
                |         |
                v         v
             BM25      Dense
             Weight    Weight
                |         |
                +----+----+
                     |
                     v
              Adaptive Fusion
                     |
                     v
                   Top-K

    Each user has an isolated Chroma collection:

        user_rag_<user_id>

    The retriever also exposes real execution metrics so that
    the frontend can visualize the actual retrieval process.
    """

    # ========================================================
    # Initialization
    # ========================================================

    def __init__(self):

        print(
            f"Loading user-RAG embedding model: "
            f"{EMBEDDING_MODEL}"
        )

        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DB_DIR)
        )

        # ----------------------------------------------------
        # BM25 cache
        # ----------------------------------------------------

        self.bm25_cache = {}

        print(
            "User-RAG embedding model loaded successfully."
        )

    # ========================================================
    # Collection
    # ========================================================

    def collection_name(self, user_id):

        if not user_id or not str(user_id).strip():

            raise ValueError(
                "user_id cannot be empty."
            )

        return (
            f"user_rag_{str(user_id).strip()}"
        )

    # ========================================================

    def get_collection(self, user_id):

        name = self.collection_name(
            user_id
        )

        try:

            collection = (
                self.client.get_collection(
                    name
                )
            )

        except Exception as error:

            raise ValueError(
                f"No document collection found "
                f"for user '{user_id}'. "
                f"Upload a document first."
            ) from error

        count = collection.count()

        if count == 0:

            raise ValueError(
                f"User '{user_id}' has no "
                f"indexed document chunks."
            )

        return collection

    # ========================================================
    # Tokenization
    # ========================================================

    def tokenize(self, text):

        return re.findall(
            r"\b[\w\+\#]+\b",
            str(text).lower()
        )

    # ========================================================
    # Load User Documents
    # ========================================================

    def load_documents(self, user_id):

        collection = self.get_collection(
            user_id
        )

        results = collection.get(
            include=[
                "documents",
                "metadatas",
            ]
        )

        documents = (
            results.get(
                "documents",
                []
            )
            or []
        )

        ids = (
            results.get(
                "ids",
                []
            )
            or []
        )

        metadatas = (
            results.get(
                "metadatas",
                []
            )
            or []
        )

        if not documents:

            raise ValueError(
                "No indexed document chunks found."
            )

        # ----------------------------------------------------
        # Defensive validation
        # ----------------------------------------------------

        if len(documents) != len(ids):

            raise ValueError(
                "User document/ID count mismatch: "
                f"{len(documents)} documents vs "
                f"{len(ids)} IDs."
            )

        if len(documents) != len(metadatas):

            raise ValueError(
                "User document/metadata count mismatch: "
                f"{len(documents)} documents vs "
                f"{len(metadatas)} metadata records."
            )

        return (
            collection,
            documents,
            ids,
            metadatas,
        )

    # ========================================================
    # BM25 Index
    # ========================================================

    def get_bm25_index(
        self,
        user_id,
        documents,
        ids,
        metadatas,
    ):

        collection_name = (
            self.collection_name(
                user_id
            )
        )

        current_count = len(
            documents
        )

        cached = self.bm25_cache.get(
            collection_name
        )

        # ----------------------------------------------------
        # Reuse cache only when document count is unchanged.
        # ----------------------------------------------------

        if (
            cached is not None
            and cached["count"] == current_count
        ):

            return cached

        print(
            f"Building user BM25 index: "
            f"{collection_name}"
        )

        tokenized_documents = [

            self.tokenize(
                document
            )

            for document in documents

        ]

        bm25 = BM25Okapi(
            tokenized_documents
        )

        cached = {

            "count":
                current_count,

            "bm25":
                bm25,

            "documents":
                documents,

            "ids":
                ids,

            "metadatas":
                metadatas,

        }

        self.bm25_cache[
            collection_name
        ] = cached

        return cached

    # ========================================================
    # Clear User Cache
    # ========================================================

    def clear_user_cache(
        self,
        user_id,
    ):

        collection_name = (
            self.collection_name(
                user_id
            )
        )

        self.bm25_cache.pop(
            collection_name,
            None
        )

    # ========================================================
    # BM25 Search
    # ========================================================

    def bm25_search(
        self,
        query,
        documents,
        ids,
        metadatas,
        user_id,
        top_k,
    ):

        index = self.get_bm25_index(

            user_id=user_id,

            documents=documents,

            ids=ids,

            metadatas=metadatas,

        )

        tokenized_query = (
            self.tokenize(
                query
            )
        )

        if not tokenized_query:

            return (
                [],
                [],
                [],
                [],
            )

        scores = (
            index["bm25"].get_scores(
                tokenized_query
            )
        )

        actual_k = min(
            top_k,
            len(documents)
        )

        if actual_k <= 0:

            return (
                [],
                [],
                [],
                [],
            )

        top_indices = (
            np.argsort(
                scores
            )[::-1][:actual_k]
        )

        result_documents = [
            documents[i]
            for i in top_indices
        ]

        result_scores = [
            float(scores[i])
            for i in top_indices
        ]

        result_ids = [
            ids[i]
            for i in top_indices
        ]

        result_metadatas = [
            metadatas[i] or {}
            for i in top_indices
        ]

        return (
            result_documents,
            result_scores,
            result_ids,
            result_metadatas,
        )

    # ========================================================
    # Dense Search
    # ========================================================

    def dense_search(
        self,
        query,
        collection,
        top_k,
    ):

        collection_count = (
            collection.count()
        )

        actual_k = min(
            top_k,
            collection_count
        )

        if actual_k <= 0:

            return (
                [],
                [],
                [],
                [],
            )

        embedding = self.model.encode(

            query,

            normalize_embeddings=False,

        ).tolist()

        results = collection.query(

            query_embeddings=[
                embedding
            ],

            n_results=actual_k,

            include=[
                "documents",
                "distances",
                "metadatas",
            ],
        )

        documents = (
            results.get(
                "documents",
                [[]]
            )[0]
            or []
        )

        distances = (
            results.get(
                "distances",
                [[]]
            )[0]
            or []
        )

        ids = (
            results.get(
                "ids",
                [[]]
            )[0]
            or []
        )

        metadatas = (
            results.get(
                "metadatas",
                [[]]
            )[0]
            or []
        )

        return (
            documents,
            distances,
            ids,
            metadatas,
        )

    # ========================================================
    # Adaptive Search
    # ========================================================

    def search(
        self,
        user_id,
        query,
        top_k=TOP_K,
    ):

        if not query or not query.strip():

            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:

            raise ValueError(
                "top_k must be greater than zero."
            )

        # ====================================================
        # Overall Retrieval Timer
        # ====================================================

        retrieval_start = time.perf_counter()

        # ====================================================
        # Load isolated user collection
        # ====================================================

        (
            collection,
            documents,
            ids,
            metadatas,
        ) = self.load_documents(
            user_id
        )

        candidate_k = min(
            CANDIDATE_K,
            len(documents)
        )

        # ====================================================
        # 1. BM25 Retrieval
        # ====================================================

        bm25_start = time.perf_counter()

        (
            bm25_docs,
            bm25_scores,
            bm25_ids,
            bm25_metadatas,
        ) = self.bm25_search(

            query=query,

            documents=documents,

            ids=ids,

            metadatas=metadatas,

            user_id=user_id,

            top_k=candidate_k,

        )

        bm25_latency_ms = (
            time.perf_counter()
            - bm25_start
        ) * 1000

        # ====================================================
        # 2. Dense Retrieval
        # ====================================================

        dense_start = time.perf_counter()

        (
            dense_docs,
            dense_distances,
            dense_ids,
            dense_metadatas,
        ) = self.dense_search(

            query=query,

            collection=collection,

            top_k=candidate_k,

        )

        dense_latency_ms = (
            time.perf_counter()
            - dense_start
        ) * 1000

        # ====================================================
        # Safety Check
        # ====================================================

        if not bm25_docs and not dense_docs:

            total_retrieval_ms = (
                time.perf_counter()
                - retrieval_start
            ) * 1000

            return {

                "user_id":
                    user_id,

                "query":
                    query,

                "weights": {
                    "bm25": 0.0,
                    "dense": 0.0,
                },

                "model":
                    None,

                "retrieval_config": {

                    "candidate_k":
                        candidate_k,

                    "final_k":
                        top_k,

                    "bm25_candidates":
                        0,

                    "dense_candidates":
                        0,

                    "fused_candidates":
                        0,
                },

                "retrieval_metrics": {

                    "bm25_ms":
                        round(
                            bm25_latency_ms,
                            2
                        ),

                    "dense_ms":
                        round(
                            dense_latency_ms,
                            2
                        ),

                    "feature_extraction_ms":
                        0.0,

                    "adaptive_prediction_ms":
                        0.0,

                    "fusion_ms":
                        0.0,

                    "total_retrieval_ms":
                        round(
                            total_retrieval_ms,
                            2
                        ),
                },

                "results":
                    [],
            }

        # ====================================================
        # 3. Retrieval Features
        # ====================================================

        feature_start = time.perf_counter()

        retrieval_features = (
            retrieval_feature_service.extract(

                bm25_scores=bm25_scores,

                dense_distances=dense_distances,

                bm25_docs=bm25_docs,

                dense_docs=dense_docs,

            )
        )

        feature_latency_ms = (
            time.perf_counter()
            - feature_start
        ) * 1000

        # ====================================================
        # 4. Adaptive Prediction
        # ====================================================

        prediction_start = time.perf_counter()

        prediction = (
            adaptive_predictor_v4.predict(

                query=query,

                retrieval_features=
                    retrieval_features,

            )
        )

        prediction_latency_ms = (
            time.perf_counter()
            - prediction_start
        ) * 1000

        bm25_weight = float(
            prediction[
                "bm25_weight"
            ]
        )

        dense_weight = float(
            prediction[
                "dense_weight"
            ]
        )

        # ====================================================
        # 5. Adaptive Fusion
        # ====================================================

        fusion_start = time.perf_counter()

        fused_results = (
            fusion_service.fuse(

                dense_docs=dense_docs,

                dense_distances=
                    dense_distances,

                dense_ids=dense_ids,

                dense_metadatas=
                    dense_metadatas,

                bm25_docs=bm25_docs,

                bm25_scores=bm25_scores,

                bm25_ids=bm25_ids,

                bm25_metadatas=
                    bm25_metadatas,

                dense_weight=
                    dense_weight,

                bm25_weight=
                    bm25_weight,

            )
        )

        fusion_latency_ms = (
            time.perf_counter()
            - fusion_start
        ) * 1000

        # ====================================================
        # 6. Final Results
        # ====================================================

        final_results = (
            fused_results[
                :candidate_k
            ]
        )

        # ====================================================
        # 7. Citation Metadata
        # ====================================================

        for rank, item in enumerate(

            final_results,

            start=1,

        ):

            metadata = (
                item.get(
                    "metadata",
                    {}
                )
                or {}
            )

            item["rank"] = rank

            item["source"] = (
                metadata.get(
                    "source"
                )
            )

            item["file_id"] = (
                metadata.get(
                    "file_id"
                )
            )

            item["file_name"] = (
                metadata.get(
                    "file_name"
                )
            )

            item["document_id"] = (
                metadata.get(
                    "document_id"
                )
            )

            item["chunk_id"] = (
                metadata.get(
                    "chunk_id"
                )
            )

            item["chunk_index"] = (
                metadata.get(
                    "chunk_index"
                )
            )

            item["citation"] = {

                "source":
                    metadata.get(
                        "source"
                    ),

                "file_id":
                    metadata.get(
                        "file_id"
                    ),

                "file_name":
                    metadata.get(
                        "file_name"
                    ),

                "document_id":
                    metadata.get(
                        "document_id"
                    ),

                "chunk_id":
                    metadata.get(
                        "chunk_id"
                    ),

                "chunk_index":
                    metadata.get(
                        "chunk_index"
                    ),
            }

        # ====================================================
        # 8. Retrieval Metrics
        # ====================================================

        total_retrieval_ms = (
            time.perf_counter()
            - retrieval_start
        ) * 1000

        retrieval_metrics = {

            "bm25_ms":
                round(
                    bm25_latency_ms,
                    2
                ),

            "dense_ms":
                round(
                    dense_latency_ms,
                    2
                ),

            "feature_extraction_ms":
                round(
                    feature_latency_ms,
                    2
                ),

            "adaptive_prediction_ms":
                round(
                    prediction_latency_ms,
                    2
                ),

            "fusion_ms":
                round(
                    fusion_latency_ms,
                    2
                ),

            "total_retrieval_ms":
                round(
                    total_retrieval_ms,
                    2
                ),
        }

        # ====================================================
        # 9. Response
        # ====================================================

        return {

            "user_id":
                user_id,

            "query":
                query,

            "weights": {

                "bm25":
                    bm25_weight,

                "dense":
                    dense_weight,
            },

            "model":
                prediction[
                    "model"
                ],

            "retrieval_config": {

                "candidate_k":
                    candidate_k,

                "final_k":
                    top_k,

                "bm25_candidates":
                    len(bm25_docs),

                "dense_candidates":
                    len(dense_docs),

                "fused_candidates":
                    len(fused_results),

                "final_results":
                    len(final_results),
            },

            "retrieval_metrics":
                retrieval_metrics,

            "results":
                final_results,
        }


# ============================================================
# Global Retriever
# ============================================================

user_adaptive_retriever = (
    UserAdaptiveRetriever()
)
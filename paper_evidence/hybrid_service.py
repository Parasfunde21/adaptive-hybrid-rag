import time

from services.retrieval_service import dense_search
from services.bm25_service import bm25_search

from services.adaptive_predictor_v4 import (
    adaptive_predictor_v4,
)

from services.retrieval_feature_service import (
    retrieval_feature_service,
)

from services.fusion_service import (
    fusion_service,
)

from services.logging_service import (
    logging_service,
)

from config.settings import (
    TOP_K,
    CANDIDATE_K,
)


# ============================================================
# Adaptive Hybrid Retriever - V4 Rich
# ============================================================

class AdaptiveHybridRetriever:

    def __init__(self):
        pass

    # ========================================================
    # Adaptive Hybrid Search
    # ========================================================

    def search(
        self,
        query,
        top_k=TOP_K,
    ):

        total_start = time.perf_counter()

        # ====================================================
        # Retrieval Configuration
        # ====================================================

        final_k = top_k
        candidate_k = CANDIDATE_K

        # ====================================================
        # 1. BM25 + Dense Retrieval
        # ====================================================

        retrieval_start = time.perf_counter()

        (
            dense_docs,
            dense_distances,
            dense_ids,
            dense_metadatas,
        ) = dense_search(
            query=query,
            top_k=candidate_k,
        )

        (
            bm25_docs,
            bm25_scores,
            bm25_ids,
            bm25_metadatas,
        ) = bm25_search(
            query=query,
            top_k=candidate_k,
        )

        retrieval_latency = (
            time.perf_counter()
            - retrieval_start
        )

        # ====================================================
        # 2. Extract Rich Retrieval Features
        # ====================================================

        retrieval_features = (
            retrieval_feature_service.extract(
                bm25_scores=bm25_scores,
                dense_distances=dense_distances,
                bm25_docs=bm25_docs,
                dense_docs=dense_docs,
            )
        )

        # ====================================================
        # 3. V4 Rich Adaptive Weight Prediction
        # ====================================================

        prediction = adaptive_predictor_v4.predict(
            query=query,
            retrieval_features=retrieval_features,
        )

        bm25_weight = prediction[
            "bm25_weight"
        ]

        dense_weight = prediction[
            "dense_weight"
        ]

        model_name = prediction[
            "model"
        ]

        # ====================================================
        # 4. Adaptive Score Fusion
        # ====================================================

        fusion_start = time.perf_counter()

        fused_results = fusion_service.fuse(
            dense_docs=dense_docs,
            dense_distances=dense_distances,
            dense_ids=dense_ids,
            dense_metadatas=dense_metadatas,
            bm25_docs=bm25_docs,
            bm25_scores=bm25_scores,
            bm25_ids=bm25_ids,
            bm25_metadatas=bm25_metadatas,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
        )

        fusion_latency = (
            time.perf_counter()
            - fusion_start
        )

        # ====================================================
        # 5. Final Results
        # ====================================================

        final_results = fused_results[:final_k]

        # ====================================================
        # 6. Add Citation Fields
        # ====================================================

        for rank, item in enumerate(
            final_results,
            start=1,
        ):

            metadata = (
                item.get(
                    "metadata",
                    {},
                )
                or {}
            )

            item["rank"] = rank

            item["source"] = metadata.get(
                "source"
            )

            item["document_id"] = metadata.get(
                "document_id"
            )

            item["chunk_id"] = metadata.get(
                "chunk_id"
            )

            item["chunk_index"] = metadata.get(
                "chunk_index"
            )

            item["citation"] = {
                "source": metadata.get(
                    "source"
                ),
                "document_id": metadata.get(
                    "document_id"
                ),
                "chunk_id": metadata.get(
                    "chunk_id"
                ),
                "chunk_index": metadata.get(
                    "chunk_index"
                ),
            }

        # ====================================================
        # 7. Total Latency
        # ====================================================

        total_latency = (
            time.perf_counter()
            - total_start
        )

        # ====================================================
        # 8. Logging
        # ====================================================

        logging_service.log_retrieval(
            query=query,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
            retrieval_features=retrieval_features,
            model_name=model_name,
            retrieval_latency=retrieval_latency,
            fusion_latency=fusion_latency,
            total_latency=total_latency,
            result_count=len(final_results),
        )

        # ====================================================
        # 9. Return Response
        # ====================================================

        return {
            "query": query,

            "weights": {
                "bm25": bm25_weight,
                "dense": dense_weight,
            },

            "model": model_name,

            "retrieval_config": {
                "candidate_k": candidate_k,
                "final_k": final_k,
                "bm25_candidates": len(bm25_docs),
                "dense_candidates": len(dense_docs),
                "fused_candidates": len(fused_results),
            },

            "latency": {
                "retrieval": round(
                    retrieval_latency,
                    6,
                ),
                "fusion": round(
                    fusion_latency,
                    6,
                ),
                "total": round(
                    total_latency,
                    6,
                ),
            },

            "results": final_results,
        }


# ============================================================
# Global Retriever
# ============================================================

retriever = AdaptiveHybridRetriever()


# ============================================================
# Public Function
# ============================================================

def hybrid_search(
    query,
    top_k=TOP_K,
):

    return retriever.search(
        query=query,
        top_k=top_k,
    )


# ============================================================
# Manual Testing
# ============================================================

if __name__ == "__main__":

    print("=" * 80)
    print("ADAPTIVE HYBRID RAG - V4 RICH")
    print("=" * 80)

    print(f"Candidate K : {CANDIDATE_K}")
    print(f"Final K     : {TOP_K}")
    print("Model       : Extra Trees V4 Rich")
    print("Type 'exit' to stop.")

    while True:

        query = input("\nQuery : ")

        if query.lower().strip() == "exit":
            break

        if not query.strip():
            print("Please enter a query.")
            continue

        try:

            response = hybrid_search(
                query=query,
                top_k=TOP_K,
            )

            print()
            print("=" * 80)
            print("RETRIEVAL CONFIGURATION")
            print("=" * 80)

            config = response[
                "retrieval_config"
            ]

            print(
                f"Candidate K       : "
                f"{config['candidate_k']}"
            )

            print(
                f"Final K           : "
                f"{config['final_k']}"
            )

            print(
                f"BM25 candidates   : "
                f"{config['bm25_candidates']}"
            )

            print(
                f"Dense candidates  : "
                f"{config['dense_candidates']}"
            )

            print(
                f"Fused candidates  : "
                f"{config['fused_candidates']}"
            )

            print()
            print("=" * 80)
            print("V4 RICH ADAPTIVE WEIGHTS")
            print("=" * 80)

            print(
                f"BM25 Weight : "
                f"{response['weights']['bm25']}"
            )

            print(
                f"Dense Weight: "
                f"{response['weights']['dense']}"
            )

            print(
                f"Model       : "
                f"{response['model']}"
            )

            print()
            print("=" * 80)
            print("LATENCY")
            print("=" * 80)

            latency = response["latency"]

            print(
                f"Retrieval : "
                f"{latency['retrieval']:.6f}s"
            )

            print(
                f"Fusion    : "
                f"{latency['fusion']:.6f}s"
            )

            print(
                f"Total     : "
                f"{latency['total']:.6f}s"
            )

            print()
            print("=" * 80)
            print("RETRIEVED DOCUMENTS")
            print("=" * 80)

            for item in response["results"]:

                print()

                print(
                    f"Rank          : "
                    f"{item.get('rank')}"
                )

                print(
                    f"Fusion Score  : "
                    f"{item.get('fusion_score', 0)}"
                )

                print(
                    f"Retrieved By  : "
                    f"{', '.join(item.get('retrieved_by', []))}"
                )

                print(
                    f"Dense Sim     : "
                    f"{item.get('dense_similarity', 0)}"
                )

                print(
                    f"BM25 Score    : "
                    f"{item.get('bm25_score', 0)}"
                )

                print(
                    f"Dense Rank    : "
                    f"{item.get('dense_rank', '-')}"
                )

                print(
                    f"BM25 Rank     : "
                    f"{item.get('bm25_rank', '-')}"
                )

                print(
                    f"Source        : "
                    f"{item.get('source')}"
                )

                print(
                    f"Document ID   : "
                    f"{item.get('document_id')}"
                )

                print(
                    f"Chunk ID      : "
                    f"{item.get('chunk_id')}"
                )

                print(
                    f"Chunk Index   : "
                    f"{item.get('chunk_index')}"
                )

                print("-" * 80)

                print(
                    item.get(
                        "document",
                        "",
                    )[:500]
                )

        except Exception as error:

            print()
            print("ERROR:")
            print(str(error))
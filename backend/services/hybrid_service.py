from services.retrieval_service import dense_search
from services.bm25_service import bm25_search

from services.adaptive_predictor_v3 import (
    adaptive_predictor_v3
)

from services.retrieval_feature_service import (
    retrieval_feature_service
)

from services.fusion_service import fusion_service


class AdaptiveHybridRetriever:

    def __init__(self):
        pass

    def search(
        self,
        query,
        top_k=10
    ):

        # =================================================
        # 1. Retrieve ONCE
        # =================================================

        dense_docs, dense_distances, _ = (
            dense_search(
                query=query,
                top_k=top_k
            )
        )

        bm25_docs, bm25_scores, _ = (
            bm25_search(
                query=query,
                top_k=top_k
            )
        )

        # =================================================
        # 2. Extract retrieval behavior
        # =================================================

        retrieval_features = (
            retrieval_feature_service.extract(

                bm25_scores=bm25_scores,

                dense_distances=dense_distances,

                bm25_docs=bm25_docs,

                dense_docs=dense_docs
            )
        )

        # =================================================
        # 3. Predict adaptive weights
        # =================================================

        prediction = (
            adaptive_predictor_v3.predict(

                query=query,

                retrieval_features=
                    retrieval_features
            )
        )

        bm25_weight = prediction[
            "bm25_weight"
        ]

        dense_weight = prediction[
            "dense_weight"
        ]

        # =================================================
        # 4. Adaptive Score Fusion
        # =================================================

        fused_results = fusion_service.fuse(

            dense_docs=dense_docs,

            dense_distances=dense_distances,

            bm25_docs=bm25_docs,

            bm25_scores=bm25_scores,

            dense_weight=dense_weight,

            bm25_weight=bm25_weight
        )

        # =================================================
        # 5. Return
        # =================================================

        return {

            "query": query,

            "weights": {

                "bm25":
                    bm25_weight,

                "dense":
                    dense_weight

            },

            "model":
                prediction["model"],

            "results":
                fused_results[:top_k]
        }


retriever = AdaptiveHybridRetriever()


def hybrid_search(
    query,
    top_k=10
):

    return retriever.search(
        query=query,
        top_k=top_k
    )


if __name__ == "__main__":

    while True:

        query = input(
            "\nQuery : "
        )

        if query.lower() == "exit":
            break

        response = hybrid_search(
            query
        )

        print()
        print("=" * 80)
        print("Adaptive Weights")
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
        print("Retrieved Documents")
        print("=" * 80)

        for rank, item in enumerate(
            response["results"],
            start=1
        ):

            print(
                f"\nRank : {rank}"
            )

            print(
                f"Fusion Score : "
                f"{item['fusion_score']}"
            )

            print(
                f"Retrieved By : "
                f"{', '.join(item['retrieved_by'])}"
            )

            print("-" * 80)

            print(
                item["document"][:500]
            )
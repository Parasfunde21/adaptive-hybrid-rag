from services.retrieval_service import dense_search
from services.bm25_service import bm25_search
from services.adaptive_weight import adaptive_predictor
from services.fusion_service import fusion_service


class AdaptiveHybridRetriever:

    def __init__(self):
        pass

    def search(
        self,
        query,
        top_k=10
    ):

        # -----------------------------
        # Predict adaptive weights
        # -----------------------------
        prediction = adaptive_predictor.predict(query)

        bm25_weight = prediction["bm25_weight"]
        dense_weight = prediction["dense_weight"]

        # -----------------------------
        # Dense Retrieval
        # -----------------------------
        dense_docs, dense_distances, _ = dense_search(
            query=query,
            top_k=top_k
        )

        # -----------------------------
        # BM25 Retrieval
        # -----------------------------
        bm25_docs, bm25_scores, _ = bm25_search(
            query=query,
            top_k=top_k
        )

        # -----------------------------
        # Adaptive Score Fusion
        # -----------------------------
        fused_results = fusion_service.fuse(

            dense_docs=dense_docs,
            dense_distances=dense_distances,

            bm25_docs=bm25_docs,
            bm25_scores=bm25_scores,

            dense_weight=dense_weight,
            bm25_weight=bm25_weight

        )

        return {

            "query": query,

            "weights": {

                "bm25": bm25_weight,

                "dense": dense_weight

            },

            "results": fused_results[:top_k]

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

        query = input("\nQuery : ")

        if query.lower() == "exit":
            break

        response = hybrid_search(query)

        print("\nAdaptive Weights")
        print("-" * 50)

        print(
            f"BM25 Weight : {response['weights']['bm25']}"
        )

        print(
            f"Dense Weight: {response['weights']['dense']}"
        )

        print("\nRetrieved Documents")
        print("=" * 100)

        for rank, item in enumerate(
            response["results"],
            start=1
        ):

            print(f"\nRank {rank}")

            print(
                f"Fusion Score : {item['score']}"
            )

            print(
                f"Retrieved By : {item['source']}"
            )

            print("-" * 100)

            print(item["document"][:500])
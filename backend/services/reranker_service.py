from sentence_transformers import CrossEncoder


class CrossEncoderReranker:

    def __init__(self):

        print("Loading Cross Encoder...")

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        print("Cross Encoder Loaded")

    def rerank(

        self,

        query,

        retrieved_documents,

        top_k=5

    ):

        if len(retrieved_documents) == 0:

            return []

        # -----------------------------
        # Build (query, document) pairs
        # -----------------------------

        pairs = [

            (query, item["document"])

            for item in retrieved_documents

        ]

        # -----------------------------
        # Cross Encoder Prediction
        # -----------------------------

        scores = self.model.predict(pairs)

        reranked = []

        for item, score in zip(

            retrieved_documents,

            scores

        ):

            new_item = item.copy()

            new_item["cross_score"] = float(score)

            reranked.append(new_item)

        # -----------------------------
        # Sort by Cross Score
        # -----------------------------

        reranked.sort(

            key=lambda x: x["cross_score"],

            reverse=True

        )

        # -----------------------------
        # Assign Cross Rank
        # -----------------------------

        for rank, item in enumerate(

            reranked,

            start=1

        ):

            item["cross_rank"] = rank

        return reranked[:top_k]


reranker = CrossEncoderReranker()


if __name__ == "__main__":

    from services.hybrid_service import hybrid_search

    while True:

        query = input("\nQuery : ")

        if query.lower() == "exit":
            break

        retrieval = hybrid_search(

            query=query,

            top_k=10

        )

        reranked = reranker.rerank(

            query=query,

            retrieved_documents=retrieval["results"],

            top_k=5

        )

        print("\nReranked Results")
        print("=" * 100)

        for item in reranked:

            print()

            print(f"Cross Rank        : {item['cross_rank']}")
            print(f"Cross Score       : {item['cross_score']:.4f}")
            print(f"Fusion Score      : {item['fusion_score']:.4f}")
            print(f"Retrieved By      : {', '.join(item['retrieved_by'])}")
            print(f"Dense Similarity  : {item['dense_similarity']}")
            print(f"BM25 Score        : {item['bm25_score']}")
            print(f"Dense Rank        : {item['dense_rank']}")
            print(f"BM25 Rank         : {item['bm25_rank']}")

            print("-" * 100)

            print(item["document"][:500])
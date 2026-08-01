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

        pairs = [

            (query, item["document"])

            for item in retrieved_documents

        ]

        scores = self.model.predict(pairs)

        ranked = []

        for item, score in zip(

            retrieved_documents,

            scores

        ):

            ranked.append({

                "document": item["document"],

                "fusion_score": item["score"],

                "cross_score": float(score),

                "source": item["source"]

            })

        ranked.sort(

            key=lambda x: x["cross_score"],

            reverse=True

        )

        return ranked[:top_k]


reranker = CrossEncoderReranker()


if __name__ == "__main__":

    from services.hybrid_service import hybrid_search

    query = input("Query : ")

    retrieved = hybrid_search(

        query,

        top_k=10

    )["results"]

    results = reranker.rerank(

        query,

        retrieved,

        top_k=5

    )

    print()

    for i, item in enumerate(

        results,

        start=1

    ):

        print("=" * 100)

        print(f"Rank : {i}")

        print(f"Cross Score : {item['cross_score']:.4f}")

        print(f"Fusion Score: {item['fusion_score']:.4f}")

        print()

        print(item["document"][:500])
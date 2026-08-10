import math


class RetrievalMetrics:

    """
    Standard Information Retrieval Metrics
    """

    @staticmethod
    def precision_at_k(
        retrieved_documents,
        relevant_documents,
        k
    ):

        retrieved = retrieved_documents[:k]

        if len(retrieved) == 0:
            return 0.0

        relevant = sum(

            1

            for document in retrieved

            if document in relevant_documents

        )

        return relevant / len(retrieved)

    @staticmethod
    def recall_at_k(
        retrieved_documents,
        relevant_documents,
        k
    ):

        if len(relevant_documents) == 0:
            return 0.0

        retrieved = retrieved_documents[:k]

        relevant = sum(

            1

            for document in retrieved

            if document in relevant_documents

        )

        return relevant / len(relevant_documents)

    @staticmethod
    def reciprocal_rank(
        retrieved_documents,
        relevant_documents
    ):

        for rank, document in enumerate(

            retrieved_documents,

            start=1

        ):

            if document in relevant_documents:

                return 1 / rank

        return 0.0

    @staticmethod
    def dcg_at_k(
        retrieved_documents,
        relevant_documents,
        k
    ):

        dcg = 0.0

        for rank, document in enumerate(

            retrieved_documents[:k],

            start=1

        ):

            if document in relevant_documents:

                dcg += 1 / math.log2(rank + 1)

        return dcg

    @staticmethod
    def ndcg_at_k(
        retrieved_documents,
        relevant_documents,
        k
    ):

        dcg = RetrievalMetrics.dcg_at_k(

            retrieved_documents,

            relevant_documents,

            k

        )

        ideal_documents = list(relevant_documents)

        idcg = RetrievalMetrics.dcg_at_k(

            ideal_documents,

            relevant_documents,

            min(k, len(relevant_documents))

        )

        if idcg == 0:

            return 0.0

        return dcg / idcg


metrics = RetrievalMetrics()

if __name__ == "__main__":

    retrieved = [

        "A",

        "B",

        "C",

        "D",

        "E"

    ]

    relevant = {

        "A",

        "C",

        "F"

    }

    print()

    print("Precision@5")

    print(

        metrics.precision_at_k(

            retrieved,

            relevant,

            5

        )

    )

    print()

    print("Recall@5")

    print(

        metrics.recall_at_k(

            retrieved,

            relevant,

            5

        )

    )

    print()

    print("MRR")

    print(

        metrics.reciprocal_rank(

            retrieved,

            relevant

        )

    )

    print()

    print("nDCG@5")

    print(

        metrics.ndcg_at_k(

            retrieved,

            relevant,

            5

        )

    )
from rank_bm25 import BM25Okapi
import numpy as np

from services.document_loader import get_all_documents


class BM25Retriever:

    def __init__(self):
        """
        Load all indexed documents and build BM25 index.
        """

        self.documents = get_all_documents()

        self.tokenized_docs = [
            doc.lower().split()
            for doc in self.documents
        ]

        self.bm25 = BM25Okapi(self.tokenized_docs)

    def search(self, query: str, top_k: int = 10):
        """
        Returns:
            documents,
            scores,
            indices
        """

        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]

        documents = [
            self.documents[i]
            for i in top_indices
        ]

        top_scores = [
            float(scores[i])
            for i in top_indices
        ]

        return (
            documents,
            top_scores,
            list(top_indices)
        )


bm25_retriever = BM25Retriever()


def bm25_search(query: str, top_k: int = 10):

    return bm25_retriever.search(
        query,
        top_k
    )


if __name__ == "__main__":

    query = input("Enter your query: ")

    docs, scores, indices = bm25_search(query)

    print("\nTop BM25 Results\n")

    for rank, (doc, score, idx) in enumerate(
        zip(docs, scores, indices),
        start=1
    ):

        print("=" * 80)

        print(f"Rank      : {rank}")

        print(f"Index     : {idx}")

        print(f"Score     : {score:.4f}")

        print("-" * 80)

        print(doc[:500])

        print()
from rank_bm25 import BM25Okapi
import numpy as np

from document_loader import get_all_documents


class BM25Retriever:
    def __init__(self):
        """
        Load all indexed documents and build the BM25 index.
        """
        self.documents = get_all_documents()

        # Tokenize each document
        self.tokenized_docs = [
            doc.lower().split()
            for doc in self.documents
        ]

        # Create BM25 Index
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def search(self, query: str, top_k: int = 5):
        """
        Perform BM25 search.

        Args:
            query: User query
            top_k: Number of documents to retrieve

        Returns:
            documents, scores
        """

        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]

        documents = [self.documents[i] for i in top_indices]
        top_scores = [float(scores[i]) for i in top_indices]

        return documents, top_scores


# Global instance
bm25_retriever = BM25Retriever()


def bm25_search(query: str, top_k: int = 5):
    """
    Reusable BM25 search function.
    """
    return bm25_retriever.search(query, top_k)


if __name__ == "__main__":

    query = input("Enter your query: ")

    docs, scores = bm25_search(query)

    print("\nTop BM25 Results\n")

    for i, (doc, score) in enumerate(zip(docs, scores), start=1):
        print("=" * 80)
        print(f"Rank : {i}")
        print(f"Score: {score:.4f}")
        print("-" * 80)
        print(doc[:500])
        print()
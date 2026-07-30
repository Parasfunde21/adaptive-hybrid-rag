from retrieval_service import dense_search
from bm25_service import bm25_search


def reciprocal_rank_fusion(dense_docs, bm25_docs, k=60):
    """
    Combine Dense Retrieval and BM25 using Reciprocal Rank Fusion (RRF).

    Args:
        dense_docs: Documents returned by dense retrieval
        bm25_docs: Documents returned by BM25 retrieval
        k: RRF constant (default = 60)

    Returns:
        List of dictionaries sorted by RRF score.
    """

    scores = {}

    # Dense Retrieval Contribution
    for rank, doc in enumerate(dense_docs):
        if doc not in scores:
            scores[doc] = {
                "score": 0.0,
                "sources": []
            }

        scores[doc]["score"] += 1 / (k + rank + 1)
        scores[doc]["sources"].append("Dense")

    # BM25 Contribution
    for rank, doc in enumerate(bm25_docs):
        if doc not in scores:
            scores[doc] = {
                "score": 0.0,
                "sources": []
            }

        scores[doc]["score"] += 1 / (k + rank + 1)
        scores[doc]["sources"].append("BM25")

    # Sort by descending score
    ranked_results = sorted(
        scores.items(),
        key=lambda x: x[1]["score"],
        reverse=True
    )

    results = []

    for document, info in ranked_results:
        results.append({
            "document": document,
            "rrf_score": round(info["score"], 6),
            "retrieved_from": ", ".join(info["sources"])
        })

    return results


def hybrid_search(query: str, top_k: int = 5):
    """
    Perform Hybrid Retrieval using:
        1. Dense Retrieval
        2. BM25 Retrieval
        3. Reciprocal Rank Fusion
    """

    dense_docs, _ = dense_search(query, top_k)
    bm25_docs, _ = bm25_search(query, top_k)

    fused_results = reciprocal_rank_fusion(
        dense_docs,
        bm25_docs
    )

    return fused_results[:top_k]


if __name__ == "__main__":

    query = input("Enter your query: ")

    results = hybrid_search(query)

    print("\nHybrid Retrieval Results\n")

    for i, result in enumerate(results, start=1):

        print("=" * 80)
        print(f"Rank : {i}")
        print(f"RRF Score : {result['rrf_score']}")
        print(f"Retrieved From : {result['retrieved_from']}")
        print("-" * 80)
        print(result["document"][:500])
        print()
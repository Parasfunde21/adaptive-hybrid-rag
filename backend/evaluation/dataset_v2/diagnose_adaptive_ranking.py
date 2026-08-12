"""
Diagnose whether Adaptive Hybrid actually changes the ranking
relative to Fixed Hybrid.

This is a diagnostic tool only.
It does NOT modify the retrieval pipeline.
"""

from services.bm25_service import bm25_search
from services.retrieval_service import dense_search
from services.fusion_service import fusion_service
from services.hybrid_service import hybrid_search


QUERY = "stock market investment"
CANDIDATE_K = 50
FINAL_K = 10


def print_result(label, results):

    print()
    print("=" * 90)
    print(label)
    print("=" * 90)

    for rank, item in enumerate(results[:FINAL_K], start=1):

        metadata = item.get("metadata") or {}

        print(
            f"{rank:>2}. "
            f"ID={item.get('id')} | "
            f"chunk={metadata.get('chunk_id')} | "
            f"dataset={metadata.get('dataset')} | "
            f"domain={metadata.get('domain')} | "
            f"score={item.get('fusion_score')}"
        )


def main():

    print("=" * 90)
    print("ADAPTIVE VS FIXED RANKING DIAGNOSTIC")
    print("=" * 90)

    print()
    print(f"Query        : {QUERY}")
    print(f"Candidate-K  : {CANDIDATE_K}")
    print(f"Final-K      : {FINAL_K}")

    # ============================================================
    # Retrieve candidate pools
    # ============================================================

    print()
    print("Retrieving BM25 candidates...")

    (
        bm25_docs,
        bm25_scores,
        bm25_ids,
        bm25_metadatas
    ) = bm25_search(
        QUERY,
        CANDIDATE_K
    )

    print(
        f"BM25 candidates: {len(bm25_docs)}"
    )

    print()
    print("Retrieving Dense candidates...")

    (
        dense_docs,
        dense_distances,
        dense_ids,
        dense_metadatas
    ) = dense_search(
        QUERY,
        CANDIDATE_K
    )

    print(
        f"Dense candidates: {len(dense_docs)}"
    )

    # ============================================================
    # Fixed Hybrid
    # ============================================================

    print()
    print("Building Fixed Hybrid ranking...")

    fixed_results = fusion_service.fuse(

        dense_docs,
        dense_distances,
        dense_ids,
        dense_metadatas,

        bm25_docs,
        bm25_scores,
        bm25_ids,
        bm25_metadatas,

        dense_weight=0.5,
        bm25_weight=0.5

    )[:FINAL_K]

    # ============================================================
    # Adaptive Hybrid
    # ============================================================

    print()
    print("Building Adaptive Hybrid ranking...")

    adaptive_output = hybrid_search(
        QUERY,
        FINAL_K
    )

    adaptive_results = adaptive_output["results"]

    adaptive_weights = adaptive_output["weights"]

    # ============================================================
    # Print rankings
    # ============================================================

    print_result(
        "FIXED HYBRID TOP 10",
        fixed_results
    )

    print_result(
        "ADAPTIVE HYBRID TOP 10",
        adaptive_results
    )

    # ============================================================
    # Compare IDs
    # ============================================================

    fixed_ids = [
        item.get("id")
        for item in fixed_results
    ]

    adaptive_ids = [
        item.get("id")
        for item in adaptive_results
    ]

    print()
    print("=" * 90)
    print("RANKING COMPARISON")
    print("=" * 90)

    print()
    print(
        f"Adaptive BM25 weight : "
        f"{adaptive_weights.get('bm25')}"
    )

    print(
        f"Adaptive Dense weight: "
        f"{adaptive_weights.get('dense')}"
    )

    print()

    print(
        "Exact ranking identical:",
        fixed_ids == adaptive_ids
    )

    print(
        "Same documents:",
        set(fixed_ids) == set(adaptive_ids)
    )

    overlap = (
        len(
            set(fixed_ids)
            &
            set(adaptive_ids)
        )
    )

    print(
        f"Top-{FINAL_K} document overlap: "
        f"{overlap}/{FINAL_K}"
    )

    # ============================================================
    # Rank movement
    # ============================================================

    fixed_position = {
        doc_id: rank
        for rank, doc_id
        in enumerate(
            fixed_ids,
            start=1
        )
    }

    adaptive_position = {
        doc_id: rank
        for rank, doc_id
        in enumerate(
            adaptive_ids,
            start=1
        )
    }

    common_ids = (
        set(fixed_position)
        &
        set(adaptive_position)
    )

    print()
    print("=" * 90)
    print("RANK MOVEMENT")
    print("=" * 90)

    moved = 0

    for doc_id in common_ids:

        fixed_rank = fixed_position[
            doc_id
        ]

        adaptive_rank = adaptive_position[
            doc_id
        ]

        if fixed_rank != adaptive_rank:

            moved += 1

            print(
                f"ID={doc_id} | "
                f"Fixed rank={fixed_rank} | "
                f"Adaptive rank={adaptive_rank}"
            )

    print()
    print(
        f"Documents that changed rank: "
        f"{moved}"
    )

    print()

    # ============================================================
    # Conclusion
    # ============================================================

    if fixed_ids == adaptive_ids:

        print(
            "RESULT: Adaptive weights changed, "
            "but the Top-K ranking is identical."
        )

        print(
            "Next action: inspect adaptive fusion "
            "score calculation."
        )

    elif set(fixed_ids) == set(adaptive_ids):

        print(
            "RESULT: Adaptive changes ranking "
            "positions, but not the Top-K document set."
        )

        print(
            "This is valid adaptive behavior."
        )

    else:

        print(
            "RESULT: Adaptive changes the retrieved "
            "document set."
        )

        print(
            "Adaptive retrieval is materially "
            "different from Fixed Hybrid."
        )


if __name__ == "__main__":
    main()
from pathlib import Path

from services.retrieval_feature_service import (
    V4_RICH_FEATURES,
    V4_RICH_FEATURE_TOP_K,
    retrieval_feature_service,
)


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

ORACLE_FILE = (
    BASE_DIR
    / "evaluation"
    / "v3_analysis"
    / "oracle_weight_sweep.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "evaluation"
    / "v4_analysis"
    / "rich_features.csv"
)


TOP_K = V4_RICH_FEATURE_TOP_K


def extract_rich_features(
    query,
    bm25_scores,
    dense_distances,
    bm25_docs,
    dense_docs,
    bm25_ids,
    dense_ids,
):
    features = retrieval_feature_service.extract(
        query=query,
        bm25_scores=bm25_scores,
        dense_distances=dense_distances,
        bm25_docs=bm25_docs,
        dense_docs=dense_docs,
        bm25_ids=bm25_ids,
        dense_ids=dense_ids,
    )

    return {
        feature: features[feature]
        for feature in V4_RICH_FEATURES
    }


# ============================================================
# Main
# ============================================================

def main():
    import pandas as pd

    from services.bm25_service import bm25_search
    from services.retrieval_service import dense_search

    print()
    print("=" * 80)
    print("BUILDING RICH V4 RETRIEVAL FEATURES")
    print("=" * 80)

    # --------------------------------------------------------
    # Load oracle dataset
    # --------------------------------------------------------

    if not ORACLE_FILE.exists():

        raise FileNotFoundError(
            f"Oracle file not found:\n{ORACLE_FILE}"
        )

    oracle = pd.read_csv(
        ORACLE_FILE
    )

    print()
    print(
        f"Queries: {len(oracle)}"
    )

    rows = []

    # --------------------------------------------------------
    # Process queries
    # --------------------------------------------------------

    for index, row in oracle.iterrows():

        query = str(
            row["query"]
        )

        # ====================================================
        # BM25
        # ====================================================

        (
            bm25_docs,
            bm25_scores,
            bm25_ids,
            bm25_metadatas
        ) = bm25_search(
            query,
            top_k=TOP_K
        )

        # ====================================================
        # Dense
        # ====================================================

        (
            dense_docs,
            dense_distances,
            dense_ids,
            dense_metadatas
        ) = dense_search(
            query,
            top_k=TOP_K
        )

        rows.append(
            {
                "query": query,
                **extract_rich_features(
                    query=query,
                    bm25_scores=bm25_scores,
                    dense_distances=dense_distances,
                    bm25_docs=bm25_docs,
                    dense_docs=dense_docs,
                    bm25_ids=bm25_ids,
                    dense_ids=dense_ids,
                ),
                "oracle_bm25_weight": float(
                    row["oracle_bm25_weight"]
                ),
            }
        )

        if (
            (index + 1) % 25 == 0
            or
            index + 1 == len(oracle)
        ):

            print(
                f"Processed "
                f"{index + 1}/"
                f"{len(oracle)}"
            )

    # ========================================================
    # Save
    # ========================================================

    result = pd.DataFrame(
        rows
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 80)
    print("RICH FEATURE DATASET CREATED")
    print("=" * 80)

    print()
    print(
        f"Rows    : {len(result)}"
    )

    print(
        f"Columns : {len(result.columns)}"
    )

    print()
    print("Columns:")

    for column in result.columns:

        print(
            f"  ✓ {column}"
        )

    print()
    print(
        f"Saved to:\n{OUTPUT_FILE}"
    )

    print()
    print("Dataset creation complete.")


if __name__ == "__main__":
    main()
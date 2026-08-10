import os
import numpy as np
import pandas as pd

from services.bm25_service import bm25_search
from services.retrieval_service import dense_search
from services.fusion_service import fusion_service


TEST_PATH = "training/ml_test.csv"

OUTPUT_PATH = (
    "evaluation/v3_analysis/oracle_weight_sweep.csv"
)

TOP_K = 10

WEIGHTS = np.round(
    np.arange(
        0.0,
        1.01,
        0.05
    ),
    2
)


# ============================================================
# Helpers
# ============================================================

def get_document_value(item):

    if isinstance(item, dict):
        return item.get(
            "document",
            ""
        )

    return str(item)


def reciprocal_rank(
    results,
    target_document
):

    for rank, item in enumerate(
        results,
        start=1
    ):

        document = get_document_value(
            item
        )

        if document == target_document:

            return 1.0 / rank

    return 0.0


def ndcg(
    results,
    target_document
):

    for rank, item in enumerate(
        results,
        start=1
    ):

        document = get_document_value(
            item
        )

        if document == target_document:

            return 1.0 / np.log2(
                rank + 1
            )

    return 0.0


def fuse_results(
    dense_docs,
    dense_distances,
    dense_ids,
    dense_metadatas,
    bm25_docs,
    bm25_scores,
    bm25_ids,
    bm25_metadatas,
    bm25_weight
):

    dense_weight = (
        1.0 - bm25_weight
    )

    results = fusion_service.fuse(

        dense_docs=dense_docs,

        dense_distances=dense_distances,

        dense_ids=dense_ids,

        dense_metadatas=dense_metadatas,

        bm25_docs=bm25_docs,

        bm25_scores=bm25_scores,

        bm25_ids=bm25_ids,

        bm25_metadatas=bm25_metadatas,

        dense_weight=dense_weight,

        bm25_weight=bm25_weight

    )

    return results[:TOP_K]


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 90)
    print("V3 ORACLE WEIGHT SWEEP")
    print("=" * 90)

    if not os.path.exists(
        TEST_PATH
    ):

        raise FileNotFoundError(
            TEST_PATH
        )

    test_df = pd.read_csv(
        TEST_PATH
    )

    print(
        f"\nQueries: {len(test_df)}"
    )

    print(
        f"Weights tested: {len(WEIGHTS)}"
    )

    rows = []

    for index, row in test_df.iterrows():

        query = str(
            row["query"]
        )

        target = str(
            row["source_document"]
        )

        # ----------------------------------------------------
        # Retrieve
        # ----------------------------------------------------

        (
            dense_docs,
            dense_distances,
            dense_ids,
            dense_metadatas
        ) = dense_search(

            query,

            top_k=TOP_K

        )

        (
            bm25_docs,
            bm25_scores,
            bm25_ids,
            bm25_metadatas
        ) = bm25_search(

            query,

            top_k=TOP_K

        )

        # ----------------------------------------------------
        # Sweep weights
        # ----------------------------------------------------

        best_weight = 0.0
        best_mrr = -1.0
        best_ndcg = -1.0
        best_rank = 999

        weight_results = []

        for bm25_weight in WEIGHTS:

            fused = fuse_results(

                dense_docs=dense_docs,

                dense_distances=dense_distances,

                dense_ids=dense_ids,

                dense_metadatas=dense_metadatas,

                bm25_docs=bm25_docs,

                bm25_scores=bm25_scores,

                bm25_ids=bm25_ids,

                bm25_metadatas=bm25_metadatas,

                bm25_weight=float(
                    bm25_weight
                )

            )

            mrr = reciprocal_rank(
                fused,
                target
            )

            ndcg_value = ndcg(
                fused,
                target
            )

            rank = (
                int(
                    round(
                        1.0 / mrr
                    )
                )
                if mrr > 0
                else 0
            )

            weight_results.append({

                "weight":
                    float(bm25_weight),

                "mrr":
                    mrr,

                "ndcg":
                    ndcg_value,

                "rank":
                    rank

            })

            # ------------------------------------------------
            # Primary objective = MRR
            # Secondary = nDCG
            # ------------------------------------------------

            if (
                mrr > best_mrr
                or
                (
                    mrr == best_mrr
                    and ndcg_value > best_ndcg
                )
            ):

                best_weight = float(
                    bm25_weight
                )

                best_mrr = mrr

                best_ndcg = ndcg_value

                best_rank = rank

        # ----------------------------------------------------
        # Find current V3 weight
        # ----------------------------------------------------

        v3_path = (
            "evaluation/v3_error_analysis.csv"
        )

        v3_df = pd.read_csv(
            v3_path
        )

        matching = v3_df[
            v3_df["query"].astype(str)
            == query
        ]

        if len(matching) > 0:

            v3_weight = float(
                matching.iloc[0][
                    "predicted_bm25_weight"
                ]
            )

            v3_mrr = float(
                matching.iloc[0][
                    "v3_mrr"
                ]
            )

        else:

            v3_weight = np.nan

            v3_mrr = np.nan

        # ----------------------------------------------------
        # Baselines
        # ----------------------------------------------------

        bm25_mrr = reciprocal_rank(
            [
                {
                    "document": doc
                }
                for doc in bm25_docs
            ],
            target
        )

        dense_mrr = reciprocal_rank(
            [
                {
                    "document": doc
                }
                for doc in dense_docs
            ],
            target
        )

        rows.append({

            "query":
                query,

            "oracle_bm25_weight":
                best_weight,

            "oracle_dense_weight":
                1.0 - best_weight,

            "oracle_mrr":
                best_mrr,

            "oracle_ndcg":
                best_ndcg,

            "oracle_rank":
                best_rank,

            "v3_bm25_weight":
                v3_weight,

            "v3_dense_weight":
                (
                    1.0 - v3_weight
                    if not np.isnan(v3_weight)
                    else np.nan
                ),

            "v3_mrr":
                v3_mrr,

            "oracle_gain_over_v3":
                (
                    best_mrr - v3_mrr
                    if not np.isnan(v3_mrr)
                    else np.nan
                ),

            "bm25_mrr":
                bm25_mrr,

            "dense_mrr":
                dense_mrr,

            "weight_error":
                (
                    abs(
                        best_weight
                        - v3_weight
                    )
                    if not np.isnan(v3_weight)
                    else np.nan
                )

        })

        if (
            (index + 1) % 10 == 0
            or
            (index + 1) == len(test_df)
        ):

            print(
                f"Processed "
                f"{index + 1}/"
                f"{len(test_df)}"
            )

    # ========================================================
    # Save
    # ========================================================

    result = pd.DataFrame(
        rows
    )

    os.makedirs(
        "evaluation/v3_analysis",
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 90)
    print("ORACLE ANALYSIS")
    print("=" * 90)

    print(
        f"\nAverage V3 MRR: "
        f"{result['v3_mrr'].mean():.4f}"
    )

    print(
        f"Average Oracle MRR: "
        f"{result['oracle_mrr'].mean():.4f}"
    )

    print(
        f"Average Oracle gain: "
        f"{result['oracle_gain_over_v3'].mean():.4f}"
    )

    print(
        f"Average V3 BM25 weight: "
        f"{result['v3_bm25_weight'].mean():.4f}"
    )

    print(
        f"Average Oracle BM25 weight: "
        f"{result['oracle_bm25_weight'].mean():.4f}"
    )

    print()

    # --------------------------------------------------------
    # Weight distribution
    # --------------------------------------------------------

    print(
        "ORACLE BM25 WEIGHT DISTRIBUTION"
    )

    print(
        result[
            "oracle_bm25_weight"
        ]
        .value_counts()
        .sort_index()
    )

    print()

    # --------------------------------------------------------
    # Queries where oracle improves V3
    # --------------------------------------------------------

    improved = result[
        result[
            "oracle_gain_over_v3"
        ] > 0
    ].sort_values(
        "oracle_gain_over_v3",
        ascending=False
    )

    print(
        "Queries where oracle improves V3:"
    )

    print(
        len(improved)
    )

    print()

    print(
        improved[
            [
                "query",
                "v3_bm25_weight",
                "oracle_bm25_weight",
                "v3_mrr",
                "oracle_mrr",
                "oracle_gain_over_v3"
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    print()

    # --------------------------------------------------------
    # BM25-winning queries
    # --------------------------------------------------------

    bm25_wins = result[
        result["bm25_mrr"]
        >
        result["dense_mrr"]
    ]

    print(
        "=" * 90
    )

    print(
        "BM25-WINNING QUERIES"
    )

    print(
        f"Count: {len(bm25_wins)}"
    )

    if len(bm25_wins) > 0:

        print(
            f"Average V3 BM25 weight: "
            f"{bm25_wins['v3_bm25_weight'].mean():.4f}"
        )

        print(
            f"Average Oracle BM25 weight: "
            f"{bm25_wins['oracle_bm25_weight'].mean():.4f}"
        )

    print()

    print(
        "=" * 90
    )

    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
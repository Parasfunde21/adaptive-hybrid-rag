import os
import time
import numpy as np
import pandas as pd

from services.bm25_service import bm25_search
from services.retrieval_service import dense_search
from services.retrieval_feature_service import retrieval_feature_service
from services.adaptive_predictor_v3 import adaptive_predictor_v3
from services.fusion_service import fusion_service


TEST_PATH = "training/ml_test.csv"
OUTPUT_PATH = "evaluation/v3_error_analysis.csv"

TOP_K = 10


# ============================================================
# Helpers
# ============================================================

def safe_float(value, default=0.0):
    try:
        value = float(value)

        if np.isnan(value):
            return default

        return value

    except Exception:
        return default


def get_relevant_rank(results, source_document):
    """
    Find the rank of the ground-truth document.
    Returns 0 when it is not retrieved.
    """

    for rank, item in enumerate(results, start=1):

        document = (
            item.get("document")
            if isinstance(item, dict)
            else item
        )

        if document == source_document:
            return rank

    return 0


def evaluate_results(results, source_document):

    rank = get_relevant_rank(
        results,
        source_document
    )

    precision = (
        1.0 / TOP_K
        if rank > 0
        else 0.0
    )

    recall = (
        1.0
        if rank > 0
        else 0.0
    )

    mrr = (
        1.0 / rank
        if rank > 0
        else 0.0
    )

    ndcg = (
        1.0 / np.log2(rank + 1)
        if rank > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "mrr": mrr,
        "ndcg": ndcg,
        "rank": rank
    }


# ============================================================
# Retrieve
# ============================================================

def retrieve(query):

    dense_docs, dense_distances, dense_ids, dense_metadatas = (
        dense_search(
            query=query,
            top_k=TOP_K
        )
    )

    bm25_docs, bm25_scores, bm25_ids, bm25_metadatas = (
        bm25_search(
            query=query,
            top_k=TOP_K
        )
    )

    return {
        "dense_docs": dense_docs,
        "dense_distances": dense_distances,
        "dense_ids": dense_ids,
        "dense_metadatas": dense_metadatas,

        "bm25_docs": bm25_docs,
        "bm25_scores": bm25_scores,
        "bm25_ids": bm25_ids,
        "bm25_metadatas": bm25_metadatas
    }


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 90)
    print("V3 ADAPTIVE RETRIEVAL ERROR ANALYSIS")
    print("=" * 90)

    if not os.path.exists(TEST_PATH):

        raise FileNotFoundError(
            f"Test dataset not found: {TEST_PATH}"
        )

    test_df = pd.read_csv(
        TEST_PATH
    )

    print(
        f"\nTest queries: {len(test_df)}"
    )

    rows = []

    for index, row in test_df.iterrows():

        query = str(
            row["query"]
        )

        source_document = str(
            row["source_document"]
        )

        print(
            f"Processed {index + 1}/{len(test_df)}",
            end="\r"
        )

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        retrieved = retrieve(
            query
        )

        dense_docs = retrieved[
            "dense_docs"
        ]

        dense_distances = retrieved[
            "dense_distances"
        ]

        bm25_docs = retrieved[
            "bm25_docs"
        ]

        bm25_scores = retrieved[
            "bm25_scores"
        ]

        # ----------------------------------------------------
        # Features
        # ----------------------------------------------------

        features = (
            retrieval_feature_service.extract(

                bm25_scores=bm25_scores,

                dense_distances=dense_distances,

                bm25_docs=bm25_docs,

                dense_docs=dense_docs

            )
        )

        # ----------------------------------------------------
        # V3 prediction
        # ----------------------------------------------------

        prediction = (
            adaptive_predictor_v3.predict(

                query=query,

                retrieval_features=features

            )
        )

        predicted_bm25 = safe_float(
            prediction.get(
                "bm25_weight",
                0.5
            )
        )

        predicted_dense = safe_float(
            prediction.get(
                "dense_weight",
                1.0 - predicted_bm25
            )
        )

        model_name = prediction.get(
            "model",
            "unknown"
        )

        # ----------------------------------------------------
        # V3 fusion
        # ----------------------------------------------------

        v3_results = (
            fusion_service.fuse(

                dense_docs=dense_docs,

                dense_distances=dense_distances,

                dense_ids=retrieved[
                    "dense_ids"
                ],

                dense_metadatas=retrieved[
                    "dense_metadatas"
                ],

                bm25_docs=bm25_docs,

                bm25_scores=bm25_scores,

                bm25_ids=retrieved[
                    "bm25_ids"
                ],

                bm25_metadatas=retrieved[
                    "bm25_metadatas"
                ],

                dense_weight=predicted_dense,

                bm25_weight=predicted_bm25

            )
        )

        v3_results = v3_results[
            :TOP_K
        ]

        # ----------------------------------------------------
        # Retrieval metrics
        # ----------------------------------------------------

        v3_metrics = evaluate_results(
            v3_results,
            source_document
        )

        # ----------------------------------------------------
        # Individual baselines
        # ----------------------------------------------------

        bm25_metrics = evaluate_results(

            [
                {
                    "document": doc
                }

                for doc in bm25_docs

            ],

            source_document
        )

        dense_metrics = evaluate_results(

            [
                {
                    "document": doc
                }

                for doc in dense_docs

            ],

            source_document
        )

        # ----------------------------------------------------
        # Determine strongest baseline
        # ----------------------------------------------------

        baseline_scores = {

            "BM25": bm25_metrics["mrr"],

            "Dense": dense_metrics["mrr"]

        }

        best_baseline = max(
            baseline_scores,
            key=baseline_scores.get
        )

        best_baseline_mrr = baseline_scores[
            best_baseline
        ]

        # ----------------------------------------------------
        # Weight error against simple oracle signal
        #
        # This is NOT the final oracle.
        # It tells us which individual retriever
        # won for this query.
        # ----------------------------------------------------

        if (
            bm25_metrics["mrr"]
            >
            dense_metrics["mrr"]
        ):

            preferred_retriever = "BM25"

        elif (
            dense_metrics["mrr"]
            >
            bm25_metrics["mrr"]
        ):

            preferred_retriever = "Dense"

        else:

            preferred_retriever = "Tie"

        # ----------------------------------------------------
        # Feature values
        # ----------------------------------------------------

        row_result = {

            "query": query,

            "source_document":
                source_document,

            "model":
                model_name,

            # V3 prediction
            "predicted_bm25_weight":
                predicted_bm25,

            "predicted_dense_weight":
                predicted_dense,

            # Preferred individual retriever
            "preferred_retriever":
                preferred_retriever,

            "best_baseline":
                best_baseline,

            # Metrics
            "bm25_mrr":
                bm25_metrics["mrr"],

            "dense_mrr":
                dense_metrics["mrr"],

            "v3_mrr":
                v3_metrics["mrr"],

            "bm25_ndcg":
                bm25_metrics["ndcg"],

            "dense_ndcg":
                dense_metrics["ndcg"],

            "v3_ndcg":
                v3_metrics["ndcg"],

            "v3_rank":
                v3_metrics["rank"],

            "best_baseline_mrr":
                best_baseline_mrr,

            "v3_mrr_gain":
                v3_metrics["mrr"]
                - best_baseline_mrr,

            # Retrieval features
            "bm25_top_score":
                safe_float(
                    features.get(
                        "bm25_top_score"
                    )
                ),

            "bm25_mean_score":
                safe_float(
                    features.get(
                        "bm25_mean_score"
                    )
                ),

            "bm25_score_gap":
                safe_float(
                    features.get(
                        "bm25_score_gap"
                    )
                ),

            "dense_top_similarity":
                safe_float(
                    features.get(
                        "dense_top_similarity"
                    )
                ),

            "dense_mean_similarity":
                safe_float(
                    features.get(
                        "dense_mean_similarity"
                    )
                ),

            "dense_similarity_gap":
                safe_float(
                    features.get(
                        "dense_similarity_gap"
                    )
                ),

            "overlap_ratio":
                safe_float(
                    features.get(
                        "retrieval_overlap_ratio"
                    )
                ),

            "rank_agreement":
                safe_float(
                    features.get(
                        "retrieval_rank_agreement"
                    )
                ),

            "bm25_result_count":
                safe_float(
                    features.get(
                        "bm25_result_count"
                    )
                ),

            "dense_result_count":
                safe_float(
                    features.get(
                        "dense_result_count"
                    )
                )
        }

        rows.append(
            row_result
        )

    print()

    # ========================================================
    # DataFrame
    # ========================================================

    result_df = pd.DataFrame(
        rows
    )

    # ========================================================
    # Sort by V3 failure
    # ========================================================

    result_df[
        "failure_severity"
    ] = (
        result_df[
            "best_baseline_mrr"
        ]
        -
        result_df[
            "v3_mrr"
        ]
    )

    result_df = result_df.sort_values(
        "failure_severity",
        ascending=False
    )

    # ========================================================
    # Save
    # ========================================================

    os.makedirs(
        "evaluation",
        exist_ok=True
    )

    result_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 90)
    print("V3 ERROR ANALYSIS SUMMARY")
    print("=" * 90)

    print(
        f"\nQueries analyzed: "
        f"{len(result_df)}"
    )

    print(
        f"V3 average MRR: "
        f"{result_df['v3_mrr'].mean():.4f}"
    )

    print(
        f"BM25 average MRR: "
        f"{result_df['bm25_mrr'].mean():.4f}"
    )

    print(
        f"Dense average MRR: "
        f"{result_df['dense_mrr'].mean():.4f}"
    )

    print()

    preferred_counts = (
        result_df[
            "preferred_retriever"
        ]
        .value_counts()
    )

    print(
        "Preferred individual retriever:"
    )

    print(
        preferred_counts
    )

    print()

    improved = (
        result_df[
            "v3_mrr_gain"
        ]
        > 0
    ).sum()

    equal = (
        result_df[
            "v3_mrr_gain"
        ]
        == 0
    ).sum()

    worse = (
        result_df[
            "v3_mrr_gain"
        ]
        < 0
    ).sum()

    print(
        f"V3 better than best individual retriever : {improved}"
    )

    print(
        f"V3 equal to best individual retriever    : {equal}"
    )

    print(
        f"V3 worse than best individual retriever   : {worse}"
    )

    print()

    print(
        "Top 15 V3 failure cases:"
    )

    display_columns = [

        "query",

        "predicted_bm25_weight",

        "preferred_retriever",

        "bm25_mrr",

        "dense_mrr",

        "v3_mrr",

        "v3_rank",

        "bm25_score_gap",

        "dense_similarity_gap",

        "overlap_ratio",

        "rank_agreement"

    ]

    print(
        result_df[
            display_columns
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

    print()

    print(
        f"Saved detailed analysis to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()
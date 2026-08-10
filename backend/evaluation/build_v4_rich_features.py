import re
from pathlib import Path

import numpy as np
import pandas as pd

from services.bm25_service import bm25_search
from services.retrieval_service import dense_search


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


TOP_K = 10


# ============================================================
# Text utilities
# ============================================================

def tokenize(text):

    return re.findall(
        r"\b\w+\b",
        str(text).lower()
    )


def query_features(query):

    tokens = tokenize(query)

    unique_tokens = set(tokens)

    if tokens:

        average_token_length = (
            sum(len(token) for token in tokens)
            / len(tokens)
        )

    else:

        average_token_length = 0.0

    return {
        "query_token_count": len(tokens),

        "query_unique_token_count":
            len(unique_tokens),

        "query_avg_token_length":
            average_token_length
    }


def lexical_coverage(
    query,
    document
):

    query_tokens = set(
        tokenize(query)
    )

    if not query_tokens:
        return 0.0

    document_tokens = set(
        tokenize(document)
    )

    matched = (
        query_tokens
        &
        document_tokens
    )

    return (
        len(matched)
        /
        len(query_tokens)
    )


# ============================================================
# BM25 statistics
# ============================================================

def calculate_bm25_features(
    scores
):

    scores = np.asarray(
        scores,
        dtype=float
    )

    if len(scores) == 0:

        return {
            "bm25_top_score": 0.0,
            "bm25_mean_score": 0.0,
            "bm25_score_std": 0.0,
            "bm25_score_gap": 0.0,
            "bm25_top_mean_ratio": 0.0
        }

    top_score = float(
        scores[0]
    )

    mean_score = float(
        np.mean(scores)
    )

    std_score = float(
        np.std(scores)
    )

    if len(scores) >= 2:

        score_gap = (
            top_score
            -
            float(scores[1])
        )

    else:

        score_gap = top_score

    if mean_score != 0:

        top_mean_ratio = (
            top_score
            /
            mean_score
        )

    else:

        top_mean_ratio = 0.0

    return {

        "bm25_top_score":
            top_score,

        "bm25_mean_score":
            mean_score,

        "bm25_score_std":
            std_score,

        "bm25_score_gap":
            float(score_gap),

        "bm25_top_mean_ratio":
            float(top_mean_ratio)
    }


# ============================================================
# Dense statistics
# ============================================================

def calculate_dense_features(
    distances
):

    distances = np.asarray(
        distances,
        dtype=float
    )

    if len(distances) == 0:

        return {

            "dense_top_similarity":
                0.0,

            "dense_mean_similarity":
                0.0,

            "dense_similarity_std":
                0.0,

            "dense_similarity_gap":
                0.0,

            "dense_top_mean_ratio":
                0.0
        }

    # Chroma distance -> similarity
    similarities = (
        1.0
        -
        distances
    )

    top_similarity = float(
        similarities[0]
    )

    mean_similarity = float(
        np.mean(similarities)
    )

    std_similarity = float(
        np.std(similarities)
    )

    if len(similarities) >= 2:

        similarity_gap = (
            top_similarity
            -
            float(similarities[1])
        )

    else:

        similarity_gap = (
            top_similarity
        )

    if mean_similarity != 0:

        top_mean_ratio = (
            top_similarity
            /
            mean_similarity
        )

    else:

        top_mean_ratio = 0.0

    return {

        "dense_top_similarity":
            top_similarity,

        "dense_mean_similarity":
            mean_similarity,

        "dense_similarity_std":
            std_similarity,

        "dense_similarity_gap":
            float(similarity_gap),

        "dense_top_mean_ratio":
            float(top_mean_ratio)
    }


# ============================================================
# Rank agreement
# ============================================================

def calculate_rank_agreement(
    bm25_ids,
    dense_ids,
    top_k
):

    bm25_rank = {
        item_id: rank
        for rank, item_id
        in enumerate(
            bm25_ids
        )
    }

    dense_rank = {
        item_id: rank
        for rank, item_id
        in enumerate(
            dense_ids
        )
    }

    common_ids = (
        set(bm25_ids)
        &
        set(dense_ids)
    )

    if len(common_ids) < 2:

        return 0.0

    differences = [

        abs(
            bm25_rank[item_id]
            -
            dense_rank[item_id]
        )

        for item_id in common_ids
    ]

    max_difference = (
        top_k - 1
    )

    if max_difference <= 0:

        return 1.0

    agreement = (
        1.0
        -
        (
            np.mean(differences)
            /
            max_difference
        )
    )

    return float(
        np.clip(
            agreement,
            0.0,
            1.0
        )
    )


# ============================================================
# Main
# ============================================================

def main():

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

        # ====================================================
        # BM25 features
        # ====================================================

        bm25_features = (
            calculate_bm25_features(
                bm25_scores
            )
        )

        # ====================================================
        # Dense features
        # ====================================================

        dense_features = (
            calculate_dense_features(
                dense_distances
            )
        )

        # ====================================================
        # Retrieval overlap
        # ====================================================

        bm25_id_set = set(
            bm25_ids
        )

        dense_id_set = set(
            dense_ids
        )

        union = (
            bm25_id_set
            |
            dense_id_set
        )

        intersection = (
            bm25_id_set
            &
            dense_id_set
        )

        if union:

            overlap_ratio = (
                len(intersection)
                /
                len(union)
            )

        else:

            overlap_ratio = 0.0

        # ====================================================
        # Rank agreement
        # ====================================================

        rank_agreement = (
            calculate_rank_agreement(
                bm25_ids,
                dense_ids,
                TOP_K
            )
        )

        # ====================================================
        # Query features
        # ====================================================

        q_features = (
            query_features(
                query
            )
        )

        # ====================================================
        # Lexical coverage
        # ====================================================

        if bm25_docs:

            bm25_coverage = (
                lexical_coverage(
                    query,
                    bm25_docs[0]
                )
            )

        else:

            bm25_coverage = 0.0

        if dense_docs:

            dense_coverage = (
                lexical_coverage(
                    query,
                    dense_docs[0]
                )
            )

        else:

            dense_coverage = 0.0

        # ====================================================
        # Final record
        # ====================================================

        record = {

            "query":
                query,

            # -----------------------------
            # BM25
            # -----------------------------

            "bm25_top_score":
                bm25_features[
                    "bm25_top_score"
                ],

            "bm25_mean_score":
                bm25_features[
                    "bm25_mean_score"
                ],

            "bm25_score_std":
                bm25_features[
                    "bm25_score_std"
                ],

            "bm25_score_gap":
                bm25_features[
                    "bm25_score_gap"
                ],

            "bm25_top_mean_ratio":
                bm25_features[
                    "bm25_top_mean_ratio"
                ],

            # -----------------------------
            # Dense
            # -----------------------------

            "dense_top_similarity":
                dense_features[
                    "dense_top_similarity"
                ],

            "dense_mean_similarity":
                dense_features[
                    "dense_mean_similarity"
                ],

            "dense_similarity_std":
                dense_features[
                    "dense_similarity_std"
                ],

            "dense_similarity_gap":
                dense_features[
                    "dense_similarity_gap"
                ],

            "dense_top_mean_ratio":
                dense_features[
                    "dense_top_mean_ratio"
                ],

            # -----------------------------
            # Agreement
            # -----------------------------

            "overlap_ratio":
                overlap_ratio,

            "rank_agreement":
                rank_agreement,

            # -----------------------------
            # Query
            # -----------------------------

            "query_token_count":
                q_features[
                    "query_token_count"
                ],

            "query_unique_token_count":
                q_features[
                    "query_unique_token_count"
                ],

            "query_avg_token_length":
                q_features[
                    "query_avg_token_length"
                ],

            # -----------------------------
            # Lexical coverage
            # -----------------------------

            "bm25_top_lexical_coverage":
                bm25_coverage,

            "dense_top_lexical_coverage":
                dense_coverage,

            # -----------------------------
            # Oracle target
            # -----------------------------

            "oracle_bm25_weight":
                float(
                    row[
                        "oracle_bm25_weight"
                    ]
                )
        }

        rows.append(
            record
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
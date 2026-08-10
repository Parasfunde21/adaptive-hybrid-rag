"""
Build V4 Oracle Weight Dataset.

Sources:

1. Oracle sweep:
   evaluation/v3_analysis/oracle_weight_sweep.csv

2. Retrieval features:
   evaluation/v3_error_analysis.csv

The two files are merged using the query.

Output:

training/oracle_weight_dataset.csv
"""

from pathlib import Path

import pandas as pd


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

FEATURE_FILE = (
    BASE_DIR
    / "evaluation"
    / "v3_error_analysis.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "training"
    / "oracle_weight_dataset.csv"
)


# ============================================================
# Required columns
# ============================================================

FEATURE_COLUMNS = [
    "bm25_score_gap",
    "dense_similarity_gap",
    "overlap_ratio",
    "rank_agreement",
]

ORACLE_COLUMNS = [
    "oracle_bm25_weight",
    "oracle_dense_weight",
    "oracle_mrr",
    "oracle_ndcg",
    "oracle_rank",
    "v3_bm25_weight",
    "v3_mrr",
    "oracle_gain_over_v3",
    "bm25_mrr",
    "dense_mrr",
    "weight_error",
]


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 80)
    print("BUILDING V4 ORACLE WEIGHT DATASET")
    print("=" * 80)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not ORACLE_FILE.exists():

        raise FileNotFoundError(
            f"Oracle file not found:\n{ORACLE_FILE}"
        )

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            f"Feature file not found:\n{FEATURE_FILE}"
        )

    print()
    print("Oracle file:")
    print(ORACLE_FILE)

    print()
    print("Feature file:")
    print(FEATURE_FILE)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    oracle_df = pd.read_csv(
        ORACLE_FILE
    )

    feature_df = pd.read_csv(
        FEATURE_FILE
    )

    print()
    print(
        f"Oracle rows   : {len(oracle_df)}"
    )

    print(
        f"Feature rows  : {len(feature_df)}"
    )

    # --------------------------------------------------------
    # Check query column
    # --------------------------------------------------------

    if "query" not in oracle_df.columns:

        raise ValueError(
            "Oracle dataset does not contain 'query'."
        )

    if "query" not in feature_df.columns:

        raise ValueError(
            "Feature dataset does not contain 'query'."
        )

    # --------------------------------------------------------
    # Check feature columns
    # --------------------------------------------------------

    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in feature_df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing retrieval features:\n"
            + str(missing_features)
        )

    # --------------------------------------------------------
    # Check oracle columns
    # --------------------------------------------------------

    missing_oracle = [
        column
        for column in ORACLE_COLUMNS
        if column not in oracle_df.columns
    ]

    if missing_oracle:

        print()
        print(
            "Warning: Missing optional oracle columns:"
        )

        print(
            missing_oracle
        )

    # --------------------------------------------------------
    # Normalize query strings
    # --------------------------------------------------------

    oracle_df["query"] = (
        oracle_df["query"]
        .astype(str)
        .str.strip()
    )

    feature_df["query"] = (
        feature_df["query"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Remove duplicate feature rows
    # --------------------------------------------------------

    feature_df = (
        feature_df
        .drop_duplicates(
            subset=["query"],
            keep="first"
        )
        .copy()
    )

    # --------------------------------------------------------
    # Select feature columns
    # --------------------------------------------------------

    features = feature_df[
        [
            "query",
            *FEATURE_COLUMNS
        ]
    ].copy()

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    dataset = oracle_df.merge(
        features,
        on="query",
        how="inner"
    )

    print()
    print(
        f"Merged rows: {len(dataset)}"
    )

    # --------------------------------------------------------
    # Detect missing queries
    # --------------------------------------------------------

    oracle_queries = set(
        oracle_df["query"]
    )

    feature_queries = set(
        features["query"]
    )

    missing_feature_queries = (
        oracle_queries
        - feature_queries
    )

    missing_oracle_queries = (
        feature_queries
        - oracle_queries
    )

    if missing_feature_queries:

        print()
        print(
            "Queries missing retrieval features:"
        )

        for query in sorted(
            missing_feature_queries
        ):

            print(
                f"  - {query}"
            )

    if missing_oracle_queries:

        print()
        print(
            "Queries missing oracle labels:"
        )

        for query in sorted(
            missing_oracle_queries
        ):

            print(
                f"  - {query}"
            )

    # --------------------------------------------------------
    # Validate merge
    # --------------------------------------------------------

    if len(dataset) == 0:

        raise ValueError(
            "Merge produced zero rows."
        )

    if len(dataset) < len(oracle_df):

        print()
        print(
            "WARNING:"
        )

        print(
            f"Oracle rows: {len(oracle_df)}"
        )

        print(
            f"Merged rows: {len(dataset)}"
        )

    # --------------------------------------------------------
    # Clean numerical values
    # --------------------------------------------------------

    numerical_columns = (
        FEATURE_COLUMNS
        + [
            "oracle_bm25_weight",
            "oracle_dense_weight",
            "oracle_mrr",
            "oracle_ndcg",
            "oracle_rank",
            "v3_bm25_weight",
            "v3_mrr",
            "oracle_gain_over_v3",
            "bm25_mrr",
            "dense_mrr",
            "weight_error",
        ]
    )

    for column in numerical_columns:

        if column in dataset.columns:

            dataset[column] = pd.to_numeric(
                dataset[column],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Remove rows with missing target/features
    # --------------------------------------------------------

    before = len(dataset)

    dataset = dataset.dropna(
        subset=[
            *FEATURE_COLUMNS,
            "oracle_bm25_weight"
        ]
    )

    after = len(dataset)

    print()
    print(
        f"Rows removed due to missing values: "
        f"{before - after}"
    )

    # --------------------------------------------------------
    # Clip oracle target
    # --------------------------------------------------------

    dataset["oracle_bm25_weight"] = (
        dataset["oracle_bm25_weight"]
        .clip(
            lower=0.0,
            upper=1.0
        )
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    dataset = (
        dataset
        .sort_values("query")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    dataset.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 80)
    print("V4 DATASET CREATED")
    print("=" * 80)

    print()
    print(
        f"Rows    : {len(dataset)}"
    )

    print(
        f"Columns : {len(dataset.columns)}"
    )

    print()
    print(
        "Features:"
    )

    for column in FEATURE_COLUMNS:

        print(
            f"  ✓ {column}"
        )

    print()
    print(
        "Target:"
    )

    print(
        "  ✓ oracle_bm25_weight"
    )

    print()
    print(
        "Oracle BM25 weight distribution:"
    )

    print(
        dataset[
            "oracle_bm25_weight"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        "Feature statistics:"
    )

    print(
        dataset[
            FEATURE_COLUMNS
            + ["oracle_bm25_weight"]
        ]
        .describe()
        .round(4)
        .to_string()
    )

    print()
    print(
        "First 10 rows:"
    )

    print(
        dataset[
            [
                "query",
                *FEATURE_COLUMNS,
                "oracle_bm25_weight"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print()
    print(
        f"Saved to:\n{OUTPUT_FILE}"
    )

    print()
    print(
        "Dataset creation complete."
    )


if __name__ == "__main__":
    main()
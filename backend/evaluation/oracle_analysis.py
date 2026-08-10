import pandas as pd


DATASET_PATH = "training/optimized_training_dataset.csv"


def analyze_oracle():

    df = pd.read_csv(DATASET_PATH)

    print()
    print("=" * 70)
    print("ORACLE WEIGHT ANALYSIS")
    print("=" * 70)

    # -------------------------------------------------
    # Basic statistics
    # -------------------------------------------------

    print()
    print("Dataset")
    print("-" * 70)

    print(
        f"Total queries       : {len(df)}"
    )

    print(
        f"Unique BM25 weights : "
        f"{df['bm25_weight'].nunique()}"
    )

    # -------------------------------------------------
    # Weight distribution
    # -------------------------------------------------

    print()
    print("BM25 Weight Distribution")
    print("-" * 70)

    distribution = (

        df["bm25_weight"]

        .value_counts()

        .sort_index()

    )

    for weight, count in distribution.items():

        percentage = (

            count /

            len(df)

        ) * 100

        print(

            f"{weight:>4.2f} : "

            f"{count:>4} queries "

            f"({percentage:>6.2f}%)"

        )

    # -------------------------------------------------
    # Dense-only cases
    # -------------------------------------------------

    dense_only = df[
        df["bm25_weight"] == 0.0
    ]

    bm25_only = df[
        df["bm25_weight"] == 1.0
    ]

    hybrid = df[
        (df["bm25_weight"] > 0.0)
        &
        (df["bm25_weight"] < 1.0)
    ]

    print()
    print("Optimization Categories")
    print("-" * 70)

    print(
        f"Dense only         : "
        f"{len(dense_only)} "
        f"({len(dense_only) / len(df) * 100:.2f}%)"
    )

    print(
        f"Hybrid             : "
        f"{len(hybrid)} "
        f"({len(hybrid) / len(df) * 100:.2f}%)"
    )

    print(
        f"BM25 only          : "
        f"{len(bm25_only)} "
        f"({len(bm25_only) / len(df) * 100:.2f}%)"
    )

    # -------------------------------------------------
    # Oracle performance
    # -------------------------------------------------

    print()
    print("Oracle Performance")
    print("-" * 70)

    print(
        f"Precision : "
        f"{df['target_precision'].mean():.4f}"
    )

    print(
        f"Recall    : "
        f"{df['target_recall'].mean():.4f}"
    )

    print(
        f"MRR       : "
        f"{df['target_mrr'].mean():.4f}"
    )

    print(
        f"nDCG      : "
        f"{df['target_ndcg'].mean():.4f}"
    )

    # -------------------------------------------------
    # Performance of non-dense queries
    # -------------------------------------------------

    print()
    print("Queries Where BM25 Was Useful")
    print("-" * 70)

    if len(hybrid) > 0:

        print(
            f"Queries requiring some BM25 "
            f"weight : {len(hybrid)}"
        )

        print(
            f"Average optimal BM25 weight : "
            f"{hybrid['bm25_weight'].mean():.4f}"
        )

        print(
            f"Average optimal Dense weight : "
            f"{hybrid['dense_weight'].mean():.4f}"
        )

        print(
            f"Average oracle MRR : "
            f"{hybrid['target_mrr'].mean():.4f}"
        )

        print(
            f"Average oracle nDCG : "
            f"{hybrid['target_ndcg'].mean():.4f}"
        )

    # -------------------------------------------------
    # Queries where BM25 was dominant
    # -------------------------------------------------

    print()
    print("BM25-Dominant Queries")
    print("-" * 70)

    dominant = df[
        df["bm25_weight"] >= 0.5
    ]

    print(
        f"Queries with BM25 >= 0.5 : "
        f"{len(dominant)}"
    )

    if len(dominant) > 0:

        print(
            f"Average BM25 weight : "
            f"{dominant['bm25_weight'].mean():.4f}"
        )

        print(
            f"Average nDCG        : "
            f"{dominant['target_ndcg'].mean():.4f}"
        )

    # -------------------------------------------------
    # Sample optimized queries
    # -------------------------------------------------

    print()
    print("Examples Where BM25 Contributed")
    print("-" * 70)

    examples = (

        df[
            df["bm25_weight"] > 0.0
        ]

        .sort_values(
            "bm25_weight",
            ascending=False
        )

        .head(15)

    )

    for _, row in examples.iterrows():

        print()
        print(
            f"Query        : "
            f"{row['query']}"
        )

        print(
            f"BM25 weight  : "
            f"{row['bm25_weight']:.2f}"
        )

        print(
            f"Dense weight : "
            f"{row['dense_weight']:.2f}"
        )

        print(
            f"MRR          : "
            f"{row['target_mrr']:.4f}"
        )

        print(
            f"nDCG         : "
            f"{row['target_ndcg']:.4f}"
        )


if __name__ == "__main__":

    analyze_oracle()
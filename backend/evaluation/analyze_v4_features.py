import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

FILE = (
    BASE_DIR
    / "training"
    / "oracle_weight_dataset.csv"
)


FEATURES = [
    "bm25_score_gap",
    "dense_similarity_gap",
    "overlap_ratio",
    "rank_agreement",
]


def main():

    print()
    print("=" * 80)
    print("V4 FEATURE ANALYSIS")
    print("=" * 80)

    df = pd.read_csv(FILE)

    print()
    print(f"Total queries: {len(df)}")

    # --------------------------------------------------
    # Define BM25 useful / not useful
    # --------------------------------------------------

    df["bm25_needed"] = (
        df["oracle_bm25_weight"] > 0
    )

    print()
    print(
        "BM25 NOT needed:",
        (~df["bm25_needed"]).sum()
    )

    print(
        "BM25 needed:",
        df["bm25_needed"].sum()
    )

    # --------------------------------------------------
    # Group statistics
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("FEATURES: BM25 NOT NEEDED")
    print("=" * 80)

    print(
        df.loc[
            ~df["bm25_needed"],
            FEATURES
        ].describe().round(4)
    )

    print()
    print("=" * 80)
    print("FEATURES: BM25 NEEDED")
    print("=" * 80)

    print(
        df.loc[
            df["bm25_needed"],
            FEATURES
        ].describe().round(4)
    )

    # --------------------------------------------------
    # Means
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("MEAN FEATURE DIFFERENCE")
    print("=" * 80)

    means = pd.DataFrame({

        "BM25_not_needed":
            df.loc[
                ~df["bm25_needed"],
                FEATURES
            ].mean(),

        "BM25_needed":
            df.loc[
                df["bm25_needed"],
                FEATURES
            ].mean()

    })

    means["difference"] = (
        means["BM25_needed"]
        -
        means["BM25_not_needed"]
    )

    print(
        means.round(4)
    )

    # --------------------------------------------------
    # Correlation
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("CORRELATION WITH ORACLE BM25 WEIGHT")
    print("=" * 80)

    correlations = (
        df[
            FEATURES
            +
            ["oracle_bm25_weight"]
        ]
        .corr()["oracle_bm25_weight"]
        .drop("oracle_bm25_weight")
        .sort_values(
            ascending=False
        )
    )

    print(
        correlations.round(4)
    )

    # --------------------------------------------------
    # BM25-heavy queries
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("BM25-HEAVY QUERIES")
    print("=" * 80)

    heavy = df[
        df["oracle_bm25_weight"] >= 0.5
    ][
        [
            "query",
            "oracle_bm25_weight"
        ]
        +
        FEATURES
    ].sort_values(
        "oracle_bm25_weight",
        ascending=False
    )

    print(
        heavy.to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    output = (
        BASE_DIR
        / "evaluation"
        / "v4_analysis"
        / "feature_analysis.csv"
    )

    means.to_csv(
        output
    )

    print()
    print(
        f"Saved:\n{output}"
    )


if __name__ == "__main__":
    main()
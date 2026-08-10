import os
import pandas as pd


INPUT_PATH = "evaluation/v3_error_analysis.csv"
OUTPUT_DIR = "evaluation/v3_analysis"


def classify_query(row):

    bm25_mrr = float(row["bm25_mrr"])
    dense_mrr = float(row["dense_mrr"])
    v3_mrr = float(row["v3_mrr"])
    bm25_weight = float(row["predicted_bm25_weight"])

    bm25_gap = float(row["bm25_score_gap"])
    dense_gap = float(row["dense_similarity_gap"])
    overlap = float(row["overlap_ratio"])
    rank_agreement = float(row["rank_agreement"])

    # --------------------------------------------------------
    # Individual retriever winner
    # --------------------------------------------------------

    if bm25_mrr > dense_mrr:
        winner = "BM25"

    elif dense_mrr > bm25_mrr:
        winner = "Dense"

    else:
        winner = "Tie"

    # --------------------------------------------------------
    # V3 status
    # --------------------------------------------------------

    best_individual = max(
        bm25_mrr,
        dense_mrr
    )

    if v3_mrr > best_individual:
        v3_status = "Better"

    elif v3_mrr < best_individual:
        v3_status = "Worse"

    else:
        v3_status = "Equal"

    # --------------------------------------------------------
    # Query type
    # --------------------------------------------------------

    if winner == "BM25":

        if bm25_weight < 0.30:
            query_type = "BM25-win-underweighted"

        elif bm25_weight < 0.50:
            query_type = "BM25-win-balanced"

        else:
            query_type = "BM25-win-correct"

    elif winner == "Dense":

        if bm25_weight > 0.70:
            query_type = "Dense-win-BM25-overweighted"

        elif bm25_weight > 0.50:
            query_type = "Dense-win-balanced"

        else:
            query_type = "Dense-win-correct"

    else:

        if bm25_weight < 0.30:
            query_type = "Tie-Dense-heavy"

        elif bm25_weight > 0.70:
            query_type = "Tie-BM25-heavy"

        else:
            query_type = "Tie-balanced"

    # --------------------------------------------------------
    # Retrieval signal
    # --------------------------------------------------------

    strong_bm25 = (
        bm25_gap >= 2.0
        and bm25_mrr > dense_mrr
    )

    strong_dense = (
        dense_gap >= 0.05
        and dense_mrr > bm25_mrr
    )

    if strong_bm25:
        signal_type = "Strong-BM25"

    elif strong_dense:
        signal_type = "Strong-Dense"

    else:
        signal_type = "Mixed"

    # --------------------------------------------------------
    # Failure severity
    # --------------------------------------------------------

    failure_amount = (
        best_individual
        - v3_mrr
    )

    if failure_amount >= 0.50:
        severity = "Severe"

    elif failure_amount > 0:
        severity = "Moderate"

    else:
        severity = "None"

    return {
        "retriever_winner": winner,
        "v3_status": v3_status,
        "query_type": query_type,
        "signal_type": signal_type,
        "failure_severity": severity,
        "failure_amount": failure_amount,
        "strong_bm25_signal": strong_bm25,
        "strong_dense_signal": strong_dense,
        "analysis_overlap_ratio": overlap,
        "analysis_rank_agreement": rank_agreement
    }


def main():

    print()
    print("=" * 90)
    print("V3 QUERY TYPE AND FAILURE ANALYSIS")
    print("=" * 90)

    if not os.path.exists(INPUT_PATH):

        raise FileNotFoundError(
            f"Missing input file: {INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH
    )

    print(
        f"Loaded queries: {len(df)}"
    )

    # --------------------------------------------------------
    # Remove duplicate columns if the source CSV has any
    # --------------------------------------------------------

    df = df.loc[
        :,
        ~df.columns.duplicated()
    ].copy()

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    classifications = []

    for _, row in df.iterrows():

        classifications.append(
            classify_query(row)
        )

    analysis_df = pd.DataFrame(
        classifications
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Reset indexes before concatenation
    # --------------------------------------------------------

    df = df.reset_index(
        drop=True
    )

    analysis_df = analysis_df.reset_index(
        drop=True
    )

    result = pd.concat(
        [
            df,
            analysis_df
        ],
        axis=1
    )

    # Final duplicate-column protection
    result = result.loc[
        :,
        ~result.columns.duplicated()
    ].copy()

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Complete analysis
    # --------------------------------------------------------

    complete_path = (
        f"{OUTPUT_DIR}/complete_analysis.csv"
    )

    result.to_csv(
        complete_path,
        index=False
    )

    # --------------------------------------------------------
    # BM25 wins
    # --------------------------------------------------------

    bm25_wins = result[
        result["retriever_winner"] == "BM25"
    ].sort_values(
        "failure_amount",
        ascending=False
    )

    bm25_path = (
        f"{OUTPUT_DIR}/bm25_win_queries.csv"
    )

    bm25_wins.to_csv(
        bm25_path,
        index=False
    )

    # --------------------------------------------------------
    # Dense wins
    # --------------------------------------------------------

    dense_wins = result[
        result["retriever_winner"] == "Dense"
    ].sort_values(
        "failure_amount",
        ascending=False
    )

    dense_path = (
        f"{OUTPUT_DIR}/dense_win_queries.csv"
    )

    dense_wins.to_csv(
        dense_path,
        index=False
    )

    # --------------------------------------------------------
    # V3 failures
    # --------------------------------------------------------

    failures = result[
        result["v3_status"] == "Worse"
    ].sort_values(
        "failure_amount",
        ascending=False
    )

    failures_path = (
        f"{OUTPUT_DIR}/v3_failures.csv"
    )

    failures.to_csv(
        failures_path,
        index=False
    )

    # --------------------------------------------------------
    # BM25 underweighted
    # --------------------------------------------------------

    bm25_underweighted = result[
        (
            result["retriever_winner"] == "BM25"
        )
        &
        (
            result["predicted_bm25_weight"] < 0.30
        )
    ].sort_values(
        "failure_amount",
        ascending=False
    )

    bm25_under_path = (
        f"{OUTPUT_DIR}/bm25_underweighted.csv"
    )

    bm25_underweighted.to_csv(
        bm25_under_path,
        index=False
    )

    # --------------------------------------------------------
    # Severe failures
    # --------------------------------------------------------

    severe = result[
        result["failure_severity"] == "Severe"
    ].sort_values(
        "failure_amount",
        ascending=False
    )

    severe_path = (
        f"{OUTPUT_DIR}/severe_failures.csv"
    )

    severe.to_csv(
        severe_path,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 90)
    print("RETRIEVER WIN DISTRIBUTION")
    print("=" * 90)

    print(
        result[
            "retriever_winner"
        ].value_counts()
    )

    print()
    print("=" * 90)
    print("V3 STATUS")
    print("=" * 90)

    print(
        result[
            "v3_status"
        ].value_counts()
    )

    print()
    print("=" * 90)
    print("QUERY TYPE DISTRIBUTION")
    print("=" * 90)

    print(
        result[
            "query_type"
        ].value_counts()
    )

    print()
    print("=" * 90)
    print("SIGNAL DISTRIBUTION")
    print("=" * 90)

    print(
        result[
            "signal_type"
        ].value_counts()
    )

    print()
    print("=" * 90)
    print("V3 FAILURES BY QUERY TYPE")
    print("=" * 90)

    print(
        failures[
            "query_type"
        ].value_counts()
    )

    # ========================================================
    # BM25 UNDERWEIGHTED
    # ========================================================

    print()
    print("=" * 90)
    print("BM25 UNDERWEIGHTED CASES")
    print("=" * 90)

    print(
        f"Total: {len(bm25_underweighted)}"
    )

    if len(bm25_underweighted) > 0:

        print()

        print(
            bm25_underweighted[
                [
                    "query",
                    "predicted_bm25_weight",
                    "bm25_mrr",
                    "dense_mrr",
                    "v3_mrr",
                    "bm25_score_gap",
                    "dense_similarity_gap",
                    "overlap_ratio",
                    "rank_agreement"
                ]
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

    # ========================================================
    # SEVERE FAILURES
    # ========================================================

    print()
    print("=" * 90)
    print("TOP SEVERE FAILURES")
    print("=" * 90)

    if len(severe) > 0:

        print()

        print(
            severe[
                [
                    "query",
                    "predicted_bm25_weight",
                    "retriever_winner",
                    "bm25_mrr",
                    "dense_mrr",
                    "v3_mrr",
                    "failure_amount",
                    "bm25_score_gap",
                    "dense_similarity_gap",
                    "overlap_ratio"
                ]
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

    else:

        print(
            "No severe failures found."
        )

    # ========================================================
    # SAVE LOCATIONS
    # ========================================================

    print()
    print("=" * 90)
    print("FILES CREATED")
    print("=" * 90)

    print(
        complete_path
    )

    print(
        bm25_path
    )

    print(
        dense_path
    )

    print(
        failures_path
    )

    print(
        bm25_under_path
    )

    print(
        severe_path
    )

    print()
    print("Analysis complete.")


if __name__ == "__main__":
    main()
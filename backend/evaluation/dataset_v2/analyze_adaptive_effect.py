from pathlib import Path
import csv
from collections import defaultdict


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    BASE_DIR
    / "evaluation"
    / "dataset_v2"
    / "results"
    / "multidomain"
)

INPUT_FILE = (
    RESULTS_DIR
    / "multidomain_query_results.csv"
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "adaptive_effect_analysis.csv"
)

SUMMARY_FILE = (
    RESULTS_DIR
    / "adaptive_effect_summary.csv"
)


# ============================================================
# HELPERS
# ============================================================

def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def rank_documents(rows):
    """
    The current benchmark CSV does not contain document IDs or
    complete ranked-document lists.

    Therefore this analysis compares query-level evaluation
    metrics between Fixed Hybrid and Adaptive Hybrid.

    It does NOT falsely claim that the document ranking changed.
    """

    return rows


# ============================================================
# LOAD RESULTS
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Result file not found:\n{INPUT_FILE}"
    )


print("=" * 80)
print("ADAPTIVE VS FIXED HYBRID ANALYSIS")
print("=" * 80)

print()
print(f"Input : {INPUT_FILE}")


query_results = defaultdict(dict)

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8",
    newline=""
) as file:

    reader = csv.DictReader(file)

    required_columns = {
        "query_id",
        "domain",
        "dataset",
        "method",
        "precision_at_10",
        "recall_at_10",
        "mrr",
        "ndcg_at_10",
        "latency",
        "bm25_weight",
        "dense_weight",
    }

    missing = (
        required_columns
        - set(reader.fieldnames or [])
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    for row in reader:

        query_id = row["query_id"]
        method = row["method"]

        query_results[query_id][method] = row


print(
    f"Queries loaded : "
    f"{len(query_results)}"
)


# ============================================================
# VERIFY METHODS
# ============================================================

methods_required = {
    "Fixed Hybrid",
    "Adaptive Hybrid",
}

for query_id, methods in query_results.items():

    missing = (
        methods_required
        - set(methods.keys())
    )

    if missing:

        raise ValueError(
            f"Query {query_id} is missing methods: "
            + ", ".join(sorted(missing))
        )


# ============================================================
# QUERY-LEVEL COMPARISON
# ============================================================

analysis_rows = []


adaptive_weight_changed = 0
adaptive_mrr_improved = 0
adaptive_mrr_equal = 0
adaptive_mrr_decreased = 0

adaptive_ndcg_improved = 0
adaptive_ndcg_equal = 0
adaptive_ndcg_decreased = 0

adaptive_precision_improved = 0
adaptive_precision_equal = 0
adaptive_precision_decreased = 0

adaptive_recall_improved = 0
adaptive_recall_equal = 0
adaptive_recall_decreased = 0


EPSILON = 1e-12


for query_id in sorted(query_results):

    fixed = query_results[query_id][
        "Fixed Hybrid"
    ]

    adaptive = query_results[query_id][
        "Adaptive Hybrid"
    ]


    domain = adaptive["domain"]
    dataset = adaptive["dataset"]


    # --------------------------------------------------------
    # Weights
    # --------------------------------------------------------

    fixed_bm25 = to_float(
        fixed["bm25_weight"]
    )

    fixed_dense = to_float(
        fixed["dense_weight"]
    )

    adaptive_bm25 = to_float(
        adaptive["bm25_weight"]
    )

    adaptive_dense = to_float(
        adaptive["dense_weight"]
    )


    weight_delta = (
        adaptive_bm25
        - fixed_bm25
    )


    weight_changed = (
        abs(weight_delta)
        > EPSILON
    )


    if weight_changed:
        adaptive_weight_changed += 1


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    fixed_precision = to_float(
        fixed["precision_at_10"]
    )

    adaptive_precision = to_float(
        adaptive["precision_at_10"]
    )

    precision_delta = (
        adaptive_precision
        - fixed_precision
    )


    fixed_recall = to_float(
        fixed["recall_at_10"]
    )

    adaptive_recall = to_float(
        adaptive["recall_at_10"]
    )

    recall_delta = (
        adaptive_recall
        - fixed_recall
    )


    fixed_mrr = to_float(
        fixed["mrr"]
    )

    adaptive_mrr = to_float(
        adaptive["mrr"]
    )

    mrr_delta = (
        adaptive_mrr
        - fixed_mrr
    )


    fixed_ndcg = to_float(
        fixed["ndcg_at_10"]
    )

    adaptive_ndcg = to_float(
        adaptive["ndcg_at_10"]
    )

    ndcg_delta = (
        adaptive_ndcg
        - fixed_ndcg
    )


    # --------------------------------------------------------
    # Precision counts
    # --------------------------------------------------------

    if precision_delta > EPSILON:

        adaptive_precision_improved += 1
        precision_result = "Improved"

    elif precision_delta < -EPSILON:

        adaptive_precision_decreased += 1
        precision_result = "Decreased"

    else:

        adaptive_precision_equal += 1
        precision_result = "Equal"


    # --------------------------------------------------------
    # Recall counts
    # --------------------------------------------------------

    if recall_delta > EPSILON:

        adaptive_recall_improved += 1
        recall_result = "Improved"

    elif recall_delta < -EPSILON:

        adaptive_recall_decreased += 1
        recall_result = "Decreased"

    else:

        adaptive_recall_equal += 1
        recall_result = "Equal"


    # --------------------------------------------------------
    # MRR counts
    # --------------------------------------------------------

    if mrr_delta > EPSILON:

        adaptive_mrr_improved += 1
        mrr_result = "Improved"

    elif mrr_delta < -EPSILON:

        adaptive_mrr_decreased += 1
        mrr_result = "Decreased"

    else:

        adaptive_mrr_equal += 1
        mrr_result = "Equal"


    # --------------------------------------------------------
    # nDCG counts
    # --------------------------------------------------------

    if ndcg_delta > EPSILON:

        adaptive_ndcg_improved += 1
        ndcg_result = "Improved"

    elif ndcg_delta < -EPSILON:

        adaptive_ndcg_decreased += 1
        ndcg_result = "Decreased"

    else:

        adaptive_ndcg_equal += 1
        ndcg_result = "Equal"


    # --------------------------------------------------------
    # Save query-level record
    # --------------------------------------------------------

    analysis_rows.append({

        "query_id":
            query_id,

        "domain":
            domain,

        "dataset":
            dataset,

        "fixed_bm25_weight":
            round(fixed_bm25, 6),

        "fixed_dense_weight":
            round(fixed_dense, 6),

        "adaptive_bm25_weight":
            round(adaptive_bm25, 6),

        "adaptive_dense_weight":
            round(adaptive_dense, 6),

        "weight_delta":
            round(weight_delta, 6),

        "weight_changed":
            weight_changed,

        "fixed_precision_at_10":
            fixed_precision,

        "adaptive_precision_at_10":
            adaptive_precision,

        "precision_delta":
            round(
                precision_delta,
                8
            ),

        "precision_result":
            precision_result,

        "fixed_recall_at_10":
            fixed_recall,

        "adaptive_recall_at_10":
            adaptive_recall,

        "recall_delta":
            round(
                recall_delta,
                8
            ),

        "recall_result":
            recall_result,

        "fixed_mrr":
            fixed_mrr,

        "adaptive_mrr":
            adaptive_mrr,

        "mrr_delta":
            round(
                mrr_delta,
                8
            ),

        "mrr_result":
            mrr_result,

        "fixed_ndcg_at_10":
            fixed_ndcg,

        "adaptive_ndcg_at_10":
            adaptive_ndcg,

        "ndcg_delta":
            round(
                ndcg_delta,
                8
            ),

        "ndcg_result":
            ndcg_result,
    })


# ============================================================
# SAVE QUERY-LEVEL ANALYSIS
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


if analysis_rows:

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=analysis_rows[0].keys()
        )

        writer.writeheader()

        writer.writerows(
            analysis_rows
        )


# ============================================================
# SUMMARY
# ============================================================

total_queries = len(
    analysis_rows
)


summary_rows = [

    {
        "metric":
            "total_queries",

        "value":
            total_queries
    },

    {
        "metric":
            "adaptive_weight_changed_queries",

        "value":
            adaptive_weight_changed
    },

    {
        "metric":
            "adaptive_weight_changed_percent",

        "value":
            round(
                adaptive_weight_changed
                / total_queries
                * 100,
                2
            )
            if total_queries
            else 0
    },

    {
        "metric":
            "mrr_improved_queries",

        "value":
            adaptive_mrr_improved
    },

    {
        "metric":
            "mrr_equal_queries",

        "value":
            adaptive_mrr_equal
    },

    {
        "metric":
            "mrr_decreased_queries",

        "value":
            adaptive_mrr_decreased
    },

    {
        "metric":
            "ndcg_improved_queries",

        "value":
            adaptive_ndcg_improved
    },

    {
        "metric":
            "ndcg_equal_queries",

        "value":
            adaptive_ndcg_equal
    },

    {
        "metric":
            "ndcg_decreased_queries",

        "value":
            adaptive_ndcg_decreased
    },

    {
        "metric":
            "precision_improved_queries",

        "value":
            adaptive_precision_improved
    },

    {
        "metric":
            "precision_equal_queries",

        "value":
            adaptive_precision_equal
    },

    {
        "metric":
            "precision_decreased_queries",

        "value":
            adaptive_precision_decreased
    },

    {
        "metric":
            "recall_improved_queries",

        "value":
            adaptive_recall_improved
    },

    {
        "metric":
            "recall_equal_queries",

        "value":
            adaptive_recall_equal
    },

    {
        "metric":
            "recall_decreased_queries",

        "value":
            adaptive_recall_decreased
    },
]


with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "metric",
            "value"
        ]
    )

    writer.writeheader()

    writer.writerows(
        summary_rows
    )


# ============================================================
# CONSOLE REPORT
# ============================================================

print()
print("=" * 80)
print("FINAL ADAPTIVE EFFECT ANALYSIS")
print("=" * 80)

print()
print(
    f"Total queries                    : "
    f"{total_queries}"
)

print(
    f"Adaptive weights changed         : "
    f"{adaptive_weight_changed}"
    f" / {total_queries}"
)

print(
    f"Weight-change percentage         : "
    f"{adaptive_weight_changed / total_queries * 100:.2f}%"
)

print()
print("MRR")
print(
    f"  Improved                       : "
    f"{adaptive_mrr_improved}"
)

print(
    f"  Equal                          : "
    f"{adaptive_mrr_equal}"
)

print(
    f"  Decreased                      : "
    f"{adaptive_mrr_decreased}"
)

print()
print("nDCG@10")
print(
    f"  Improved                       : "
    f"{adaptive_ndcg_improved}"
)

print(
    f"  Equal                          : "
    f"{adaptive_ndcg_equal}"
)

print(
    f"  Decreased                      : "
    f"{adaptive_ndcg_decreased}"
)

print()
print("Precision@10")
print(
    f"  Improved                       : "
    f"{adaptive_precision_improved}"
)

print(
    f"  Equal                          : "
    f"{adaptive_precision_equal}"
)

print(
    f"  Decreased                      : "
    f"{adaptive_precision_decreased}"
)

print()
print("Recall@10")
print(
    f"  Improved                       : "
    f"{adaptive_recall_improved}"
)

print(
    f"  Equal                          : "
    f"{adaptive_recall_equal}"
)

print(
    f"  Decreased                      : "
    f"{adaptive_recall_decreased}"
)

print()
print("=" * 80)

print(
    f"Saved query analysis :\n{OUTPUT_FILE}"
)

print(
    f"Saved summary        :\n{SUMMARY_FILE}"
)

print("=" * 80)
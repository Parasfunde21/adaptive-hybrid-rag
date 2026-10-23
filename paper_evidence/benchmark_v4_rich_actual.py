from pathlib import Path
import sys

# Add backend directory to Python path
BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import csv
import json
import math
import time
from collections import defaultdict

from services.bm25_service import bm25_search
from services.retrieval_service import dense_search
from services.adaptive_predictor_v4 import adaptive_predictor_v4
from services.retrieval_feature_service import retrieval_feature_service
from services.fusion_service import fusion_service

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

BENCHMARK_FILE = (
    BASE_DIR
    / "evaluation"
    / "dataset_v2"
    / "benchmark"
    / "benchmark_queries.jsonl"
)

RESULTS_DIR = (
    BASE_DIR
    / "evaluation"
    / "dataset_v2"
    / "results"
    / "v4_actual"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Configuration
# ============================================================

TOP_K = 10
CANDIDATE_K = 50

FIXED_BM25_WEIGHT = 0.50
FIXED_DENSE_WEIGHT = 0.50


# ============================================================
# Load benchmark
# ============================================================

def load_queries():

    rows = []

    with open(
        BENCHMARK_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            rows.append(
                json.loads(line)
            )

    return rows


# ============================================================
# Relevance
# ============================================================

def build_relevance_map(row):

    relevance = {}

    for item in row.get(
        "relevant_documents",
        [],
    ):

        document_id = str(
            item["document_id"]
        )

        relevance[document_id] = int(
            item.get(
                "relevance",
                1,
            )
        )

    return relevance


def get_document_id(item):

    metadata = (
        item.get(
            "metadata",
            {},
        )
        or {}
    )

    value = (
        metadata.get("document_id")
        or item.get("document_id")
    )

    if value is None:
        return None

    return str(value)


# ============================================================
# Metrics
# ============================================================

def precision_at_k(results, relevance, k):

    retrieved = results[:k]

    if not retrieved:
        return 0.0

    relevant = 0

    for item in retrieved:

        document_id = get_document_id(item)

        if (
            document_id in relevance
            and relevance[document_id] > 0
        ):
            relevant += 1

    return relevant / k


def recall_at_k(results, relevance, k):

    if not relevance:
        return 0.0

    retrieved = results[:k]

    found = set()

    for item in retrieved:

        document_id = get_document_id(item)

        if (
            document_id in relevance
            and relevance[document_id] > 0
        ):
            found.add(document_id)

    total_relevant = sum(
        1
        for value in relevance.values()
        if value > 0
    )

    if total_relevant == 0:
        return 0.0

    return len(found) / total_relevant


def mrr(results, relevance):

    for rank, item in enumerate(
        results,
        start=1,
    ):

        document_id = get_document_id(item)

        if (
            document_id in relevance
            and relevance[document_id] > 0
        ):
            return 1.0 / rank

    return 0.0


def dcg(results, relevance, k):

    score = 0.0

    for rank, item in enumerate(
        results[:k],
        start=1,
    ):

        document_id = get_document_id(item)

        rel = relevance.get(
            document_id,
            0,
        )

        score += (
            (2 ** rel - 1)
            / math.log2(rank + 1)
        )

    return score


def ndcg_at_k(results, relevance, k):

    if not relevance:
        return 0.0

    actual = dcg(
        results,
        relevance,
        k,
    )

    ideal_relevances = sorted(
        [
            value
            for value in relevance.values()
            if value > 0
        ],
        reverse=True,
    )[:k]

    ideal = 0.0

    for rank, rel in enumerate(
        ideal_relevances,
        start=1,
    ):

        ideal += (
            (2 ** rel - 1)
            / math.log2(rank + 1)
        )

    if ideal == 0:
        return 0.0

    return actual / ideal


# ============================================================
# Convert retrieval result to common format
# ============================================================

def normalize_results(
    results,
    top_k=TOP_K,
):

    output = []

    for item in results[:top_k]:

        metadata = (
            item.get(
                "metadata",
                {},
            )
            or {}
        )

        output.append(
            {
                "document_id":
                    get_document_id(item),

                "score":
                    item.get(
                        "fusion_score",
                        item.get(
                            "score",
                            0.0,
                        ),
                    ),

                "metadata":
                    metadata,
            }
        )

    return output


# ============================================================
# RRF
# ============================================================

def reciprocal_rank_fusion(
    dense_ids,
    bm25_ids,
    k=60,
):

    scores = defaultdict(float)

    metadata = {}

    for rank, document_id in enumerate(
        dense_ids,
        start=1,
    ):

        document_id = str(
            document_id
        )

        scores[document_id] += (
            1.0
            / (k + rank)
        )

    for rank, document_id in enumerate(
        bm25_ids,
        start=1,
    ):

        document_id = str(
            document_id
        )

        scores[document_id] += (
            1.0
            / (k + rank)
        )

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        {
            "document_id": document_id,
            "score": score,
            "metadata": metadata.get(
                document_id,
                {},
            ),
        }
        for document_id, score in ranked[:TOP_K]
    ]


# ============================================================
# Evaluation
# ============================================================

def evaluate():

    queries = load_queries()

    print("=" * 80)
    print("V4 RICH ACTUAL BENCHMARK")
    print("=" * 80)
    print(
        f"Queries      : {len(queries)}"
    )
    print(
        f"Candidate K  : {CANDIDATE_K}"
    )
    print(
        f"Final K      : {TOP_K}"
    )
    print(
        "Adaptive     : Extra Trees V4 Rich"
    )
    print("=" * 80)

    methods = [
        "BM25",
        "Dense",
        "Fixed Hybrid",
        "Adaptive Hybrid",
    ]

    totals = {
        method: {
            "precision": 0.0,
            "recall": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
            "latency": 0.0,
            "queries": 0,
        }
        for method in methods
    }

    domain_totals = defaultdict(
        lambda: {
            method: {
                "precision": 0.0,
                "recall": 0.0,
                "mrr": 0.0,
                "ndcg": 0.0,
                "latency": 0.0,
                "queries": 0,
            }
            for method in methods
        }
    )

    comparison_rows = []

    weight_rows = []

    for index, row in enumerate(
        queries,
        start=1,
    ):

        query = row["query"]
        domain = row.get(
            "domain",
            "unknown",
        )

        relevance = build_relevance_map(
            row
        )

        print(
            f"\rProcessing "
            f"{index}/{len(queries)}",
            end="",
            flush=True,
        )

        # ====================================================
        # Retrieval
        # ====================================================

        start = time.perf_counter()

        (
            dense_docs,
            dense_distances,
            dense_ids,
            dense_metadatas,
        ) = dense_search(
            query=query,
            top_k=CANDIDATE_K,
        )

        dense_latency = (
            time.perf_counter()
            - start
        )

        start = time.perf_counter()

        (
            bm25_docs,
            bm25_scores,
            bm25_ids,
            bm25_metadatas,
        ) = bm25_search(
            query=query,
            top_k=CANDIDATE_K,
        )

        bm25_latency = (
            time.perf_counter()
            - start
        )

        # ====================================================
        # Normalize IDs
        # ====================================================

        dense_ids = [
            str(x)
            for x in dense_ids
        ]

        bm25_ids = [
            str(x)
            for x in bm25_ids
        ]

        # ====================================================
        # BM25
        # ====================================================

        bm25_results = []

        for rank, document_id in enumerate(
            bm25_ids[:TOP_K],
            start=1,
        ):

            metadata = {}

            if rank - 1 < len(
                bm25_metadatas
            ):
                metadata = (
                    bm25_metadatas[
                        rank - 1
                    ]
                    or {}
                )

            if not metadata:
                metadata = {
                    "document_id":
                        document_id
                }

            bm25_results.append(
                {
                    "document_id":
                        document_id,
                    "score":
                        bm25_scores[
                            rank - 1
                        ],
                    "metadata":
                        metadata,
                }
            )

        # ====================================================
        # Dense
        # ====================================================

        dense_results = []

        for rank, document_id in enumerate(
            dense_ids[:TOP_K],
            start=1,
        ):

            metadata = {}

            if rank - 1 < len(
                dense_metadatas
            ):
                metadata = (
                    dense_metadatas[
                        rank - 1
                    ]
                    or {}
                )

            if not metadata:
                metadata = {
                    "document_id":
                        document_id
                }

            dense_results.append(
                {
                    "document_id":
                        document_id,
                    "score":
                        0.0,
                    "metadata":
                        metadata,
                }
            )

        # ====================================================
        # Features + V4 Prediction
        # ====================================================

        features = (
            retrieval_feature_service.extract(
                bm25_scores=bm25_scores,
                dense_distances=dense_distances,
                bm25_docs=bm25_docs,
                dense_docs=dense_docs,
            )
        )

        prediction = (
            adaptive_predictor_v4.predict(
                query=query,
                retrieval_features=features,
            )
        )

        bm25_weight = prediction[
            "bm25_weight"
        ]

        dense_weight = prediction[
            "dense_weight"
        ]

        weight_rows.append(
            {
                "query_id":
                    row["query_id"],
                "domain":
                    domain,
                "bm25_weight":
                    bm25_weight,
                "dense_weight":
                    dense_weight,
            }
        )

        # ====================================================
        # Fixed Hybrid
        # ====================================================

        fixed_fused = (
            fusion_service.fuse(
                dense_docs=dense_docs,
                dense_distances=dense_distances,
                dense_ids=dense_ids,
                dense_metadatas=dense_metadatas,
                bm25_docs=bm25_docs,
                bm25_scores=bm25_scores,
                bm25_ids=bm25_ids,
                bm25_metadatas=bm25_metadatas,
                dense_weight=FIXED_DENSE_WEIGHT,
                bm25_weight=FIXED_BM25_WEIGHT,
            )
        )

        fixed_results = normalize_results(
            fixed_fused
        )

        # ====================================================
        # Adaptive V4 Hybrid
        # ====================================================

        adaptive_fused = (
            fusion_service.fuse(
                dense_docs=dense_docs,
                dense_distances=dense_distances,
                dense_ids=dense_ids,
                dense_metadatas=dense_metadatas,
                bm25_docs=bm25_docs,
                bm25_scores=bm25_scores,
                bm25_ids=bm25_ids,
                bm25_metadatas=bm25_metadatas,
                dense_weight=dense_weight,
                bm25_weight=bm25_weight,
            )
        )

        adaptive_results = normalize_results(
            adaptive_fused
        )

        # ====================================================
        # Metrics
        # ====================================================

        result_sets = {
            "BM25": bm25_results,
            "Dense": dense_results,
            "Fixed Hybrid": fixed_results,
            "Adaptive Hybrid": adaptive_results,
        }

        query_metrics = {}

        for method, results in result_sets.items():

            if method == "BM25":
                latency = bm25_latency

            elif method == "Dense":
                latency = dense_latency

            else:
                latency = (
                    dense_latency
                    + bm25_latency
                )

            p = precision_at_k(
                results,
                relevance,
                TOP_K,
            )

            r = recall_at_k(
                results,
                relevance,
                TOP_K,
            )

            m = mrr(
                results,
                relevance,
            )

            n = ndcg_at_k(
                results,
                relevance,
                TOP_K,
            )

            totals[method][
                "precision"
            ] += p

            totals[method][
                "recall"
            ] += r

            totals[method][
                "mrr"
            ] += m

            totals[method][
                "ndcg"
            ] += n

            totals[method][
                "latency"
            ] += latency

            totals[method][
                "queries"
            ] += 1

            domain_totals[
                domain
            ][method][
                "precision"
            ] += p

            domain_totals[
                domain
            ][method][
                "recall"
            ] += r

            domain_totals[
                domain
            ][method][
                "mrr"
            ] += m

            domain_totals[
                domain
            ][method][
                "ndcg"
            ] += n

            domain_totals[
                domain
            ][method][
                "latency"
            ] += latency

            domain_totals[
                domain
            ][method][
                "queries"
            ] += 1

            query_metrics[
                method
            ] = {
                "precision":
                    p,
                "recall":
                    r,
                "mrr":
                    m,
                "ndcg":
                    n,
            }

        comparison_rows.append(
            {
                "query_id":
                    row["query_id"],
                "domain":
                    domain,
                "bm25_weight":
                    bm25_weight,
                "dense_weight":
                    dense_weight,

                "fixed_mrr":
                    query_metrics[
                        "Fixed Hybrid"
                    ]["mrr"],

                "adaptive_mrr":
                    query_metrics[
                        "Adaptive Hybrid"
                    ]["mrr"],

                "fixed_ndcg":
                    query_metrics[
                        "Fixed Hybrid"
                    ]["ndcg"],

                "adaptive_ndcg":
                    query_metrics[
                        "Adaptive Hybrid"
                    ]["ndcg"],

                "fixed_precision":
                    query_metrics[
                        "Fixed Hybrid"
                    ]["precision"],

                "adaptive_precision":
                    query_metrics[
                        "Adaptive Hybrid"
                    ]["precision"],

                "fixed_recall":
                    query_metrics[
                        "Fixed Hybrid"
                    ]["recall"],

                "adaptive_recall":
                    query_metrics[
                        "Adaptive Hybrid"
                    ]["recall"],
            }
        )

    print("\n")

    # ========================================================
    # Summary
    # ========================================================

    summary_path = (
        RESULTS_DIR
        / "v4_actual_summary.csv"
    )

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "method",
                "queries",
                "precision_at_10",
                "recall_at_10",
                "mrr",
                "ndcg_at_10",
                "latency_seconds",
            ]
        )

        for method in methods:

            data = totals[method]
            count = data["queries"]

            writer.writerow(
                [
                    method,
                    count,
                    data["precision"] / count,
                    data["recall"] / count,
                    data["mrr"] / count,
                    data["ndcg"] / count,
                    data["latency"] / count,
                ]
            )

    # ========================================================
    # Domain Summary
    # ========================================================

    domain_path = (
        RESULTS_DIR
        / "v4_actual_domain_summary.csv"
    )

    with open(
        domain_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "domain",
                "method",
                "queries",
                "precision_at_10",
                "recall_at_10",
                "mrr",
                "ndcg_at_10",
                "latency_seconds",
            ]
        )

        for domain in sorted(
            domain_totals
        ):

            for method in methods:

                data = domain_totals[
                    domain
                ][method]

                count = data["queries"]

                writer.writerow(
                    [
                        domain,
                        method,
                        count,
                        data["precision"] / count,
                        data["recall"] / count,
                        data["mrr"] / count,
                        data["ndcg"] / count,
                        data["latency"] / count,
                    ]
                )

    # ========================================================
    # Query Comparison
    # ========================================================

    comparison_path = (
        RESULTS_DIR
        / "v4_actual_query_comparison.csv"
    )

    with open(
        comparison_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        if comparison_rows:

            writer = csv.DictWriter(
                file,
                fieldnames=
                    comparison_rows[0].keys(),
            )

            writer.writeheader()
            writer.writerows(
                comparison_rows
            )

    # ========================================================
    # Weight Distribution
    # ========================================================

    weights_path = (
        RESULTS_DIR
        / "v4_actual_weights.csv"
    )

    with open(
        weights_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "query_id",
                "domain",
                "bm25_weight",
                "dense_weight",
            ],
        )

        writer.writeheader()
        writer.writerows(
            weight_rows
        )

    # ========================================================
    # Improvement Analysis
    # ========================================================

    improved_mrr = 0
    equal_mrr = 0
    decreased_mrr = 0

    improved_ndcg = 0
    equal_ndcg = 0
    decreased_ndcg = 0

    improved_precision = 0
    equal_precision = 0
    decreased_precision = 0

    improved_recall = 0
    equal_recall = 0
    decreased_recall = 0

    changed_weights = 0

    for row in comparison_rows:

        if abs(
            row["bm25_weight"]
            - FIXED_BM25_WEIGHT
        ) > 1e-9:

            changed_weights += 1

        if (
            row["adaptive_mrr"]
            > row["fixed_mrr"]
        ):
            improved_mrr += 1

        elif (
            row["adaptive_mrr"]
            < row["fixed_mrr"]
        ):
            decreased_mrr += 1

        else:
            equal_mrr += 1

        if (
            row["adaptive_ndcg"]
            > row["fixed_ndcg"]
        ):
            improved_ndcg += 1

        elif (
            row["adaptive_ndcg"]
            < row["fixed_ndcg"]
        ):
            decreased_ndcg += 1

        else:
            equal_ndcg += 1

        if (
            row["adaptive_precision"]
            > row["fixed_precision"]
        ):
            improved_precision += 1

        elif (
            row["adaptive_precision"]
            < row["fixed_precision"]
        ):
            decreased_precision += 1

        else:
            equal_precision += 1

        if (
            row["adaptive_recall"]
            > row["fixed_recall"]
        ):
            improved_recall += 1

        elif (
            row["adaptive_recall"]
            < row["fixed_recall"]
        ):
            decreased_recall += 1

        else:
            equal_recall += 1

    improvement_path = (
        RESULTS_DIR
        / "v4_actual_improvement_summary.csv"
    )

    with open(
        improvement_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "metric",
                "value",
            ]
        )

        writer.writerow(
            [
                "total_queries",
                len(comparison_rows),
            ]
        )

        writer.writerow(
            [
                "adaptive_weight_changed_queries",
                changed_weights,
            ]
        )

        writer.writerow(
            [
                "adaptive_weight_changed_percent",
                (
                    changed_weights
                    / len(comparison_rows)
                    * 100
                ),
            ]
        )

        writer.writerow(
            [
                "mrr_improved_queries",
                improved_mrr,
            ]
        )

        writer.writerow(
            [
                "mrr_equal_queries",
                equal_mrr,
            ]
        )

        writer.writerow(
            [
                "mrr_decreased_queries",
                decreased_mrr,
            ]
        )

        writer.writerow(
            [
                "ndcg_improved_queries",
                improved_ndcg,
            ]
        )

        writer.writerow(
            [
                "ndcg_equal_queries",
                equal_ndcg,
            ]
        )

        writer.writerow(
            [
                "ndcg_decreased_queries",
                decreased_ndcg,
            ]
        )

        writer.writerow(
            [
                "precision_improved_queries",
                improved_precision,
            ]
        )

        writer.writerow(
            [
                "precision_equal_queries",
                equal_precision,
            ]
        )

        writer.writerow(
            [
                "precision_decreased_queries",
                decreased_precision,
            ]
        )

        writer.writerow(
            [
                "recall_improved_queries",
                improved_recall,
            ]
        )

        writer.writerow(
            [
                "recall_equal_queries",
                equal_recall,
            ]
        )

        writer.writerow(
            [
                "recall_decreased_queries",
                decreased_recall,
            ]
        )

    # ========================================================
    # Console Summary
    # ========================================================

    print("=" * 80)
    print("V4 RICH BENCHMARK COMPLETE")
    print("=" * 80)

    print()

    print(
        f"{'Method':<20}"
        f"{'P@10':>12}"
        f"{'R@10':>12}"
        f"{'MRR':>12}"
        f"{'NDCG@10':>12}"
        f"{'Latency':>12}"
    )

    print("-" * 80)

    for method in methods:

        data = totals[method]
        count = data["queries"]

        print(
            f"{method:<20}"
            f"{data['precision']/count:>12.4f}"
            f"{data['recall']/count:>12.4f}"
            f"{data['mrr']/count:>12.4f}"
            f"{data['ndcg']/count:>12.4f}"
            f"{data['latency']/count:>12.4f}"
        )

    print()
    print("=" * 80)
    print("ADAPTIVE EFFECT")
    print("=" * 80)

    print(
        f"Weight changed : "
        f"{changed_weights}/{len(comparison_rows)}"
    )

    print(
        f"MRR improved   : "
        f"{improved_mrr}"
    )

    print(
        f"MRR equal      : "
        f"{equal_mrr}"
    )

    print(
        f"MRR decreased  : "
        f"{decreased_mrr}"
    )

    print(
        f"NDCG improved  : "
        f"{improved_ndcg}"
    )

    print(
        f"NDCG equal     : "
        f"{equal_ndcg}"
    )

    print(
        f"NDCG decreased : "
        f"{decreased_ndcg}"
    )

    print()
    print("Output:")
    print(summary_path)
    print(domain_path)
    print(comparison_path)
    print(weights_path)
    print(improvement_path)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    evaluate()
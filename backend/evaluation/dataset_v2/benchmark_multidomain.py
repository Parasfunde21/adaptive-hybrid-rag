from pathlib import Path
import argparse
import csv
import json
import math
import time
from collections import defaultdict


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

BENCHMARK_DIR = (
    BASE_DIR
    / "evaluation"
    / "dataset_v2"
    / "benchmark"
)

RESULTS_DIR = (
    BASE_DIR
    / "evaluation"
    / "dataset_v2"
    / "results"
    / "multidomain"
)

BENCHMARK_FILE = (
    BENCHMARK_DIR
    / "benchmark_queries.jsonl"
)


# ============================================================
# Current project services
# ============================================================

from services.bm25_service import (
    bm25_search
)

from services.retrieval_service import (
    dense_search
)

from services.adaptive_predictor_v3 import (
    adaptive_predictor_v3
)

from services.retrieval_feature_service import (
    retrieval_feature_service
)

from services.fusion_service import (
    fusion_service
)


# ============================================================
# Configuration
# ============================================================

DEFAULT_TOP_K = 10

# Retrieve a larger candidate pool before fusion.
# Final evaluation is still performed at Top-K = 10.
CANDIDATE_K = 50

RRF_K = 60

FIXED_BM25_WEIGHT = 0.50
FIXED_DENSE_WEIGHT = 0.50

METHOD_ORDER = [
    "BM25",
    "Dense",
    "RRF",
    "Fixed Hybrid",
    "Adaptive Hybrid",
]


# ============================================================
# JSONL
# ============================================================

def load_jsonl(path):
    rows = []

    with open(
        path,
        "r",
        encoding="utf-8"
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
# Dataset-aware keys
# ============================================================

def result_document_key(
    item,
    default_dataset=None
):
    """
    Use dataset:document_id as the evaluation key.

    This is important because the Chroma collection contains
    multiple BEIR datasets.
    """

    metadata = (
        item.get(
            "metadata",
            {}
        )
        or {}
    )

    dataset = (
        metadata.get(
            "dataset"
        )
        or default_dataset
    )

    document_id = metadata.get(
        "document_id"
    )

    if dataset and document_id:
        return (
            f"{dataset}:"
            f"{document_id}"
        )

    if document_id:
        return str(document_id)

    chunk_id = metadata.get(
        "chunk_id"
    )

    if chunk_id:
        return str(chunk_id)

    item_id = item.get(
        "id"
    )

    if item_id:
        return str(item_id)

    return ""


def qrel_document_key(
    dataset,
    document_id
):
    return (
        f"{dataset}:"
        f"{document_id}"
    )


# ============================================================
# Raw result conversion
# ============================================================

def make_results(
    documents,
    ids,
    metadatas,
    scores=None
):
    results = []

    documents = documents or []
    ids = ids or []
    metadatas = metadatas or []
    scores = scores or []

    for index, document in enumerate(
        documents
    ):

        metadata = {}

        if index < len(metadatas):

            metadata = (
                metadatas[index]
                or {}
            )

        item_id = ""

        if index < len(ids):

            item_id = ids[index]

        score = None

        if index < len(scores):

            score = scores[index]

        results.append(
            {
                "document":
                    document,

                "id":
                    item_id,

                "metadata":
                    metadata,

                "score":
                    score
            }
        )

    return results


# ============================================================
# RRF
# ============================================================

def reciprocal_rank_fusion(
    bm25_results,
    dense_results,
    top_k=DEFAULT_TOP_K
):
    fused = {}

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    for rank, item in enumerate(
        bm25_results,
        start=1
    ):

        key = result_document_key(
            item
        )

        if not key:
            continue

        if key not in fused:

            fused[key] = {
                "document":
                    item["document"],

                "id":
                    item["id"],

                "metadata":
                    item["metadata"],

                "rrf_score":
                    0.0,

                "retrieved_by":
                    []
            }

        fused[key][
            "rrf_score"
        ] += (
            1.0
            /
            (
                RRF_K
                + rank
            )
        )

        if "BM25" not in fused[key][
            "retrieved_by"
        ]:

            fused[key][
                "retrieved_by"
            ].append(
                "BM25"
            )

    # --------------------------------------------------------
    # Dense
    # --------------------------------------------------------

    for rank, item in enumerate(
        dense_results,
        start=1
    ):

        key = result_document_key(
            item
        )

        if not key:
            continue

        if key not in fused:

            fused[key] = {
                "document":
                    item["document"],

                "id":
                    item["id"],

                "metadata":
                    item["metadata"],

                "rrf_score":
                    0.0,

                "retrieved_by":
                    []
            }

        fused[key][
            "rrf_score"
        ] += (
            1.0
            /
            (
                RRF_K
                + rank
            )
        )

        if "Dense" not in fused[key][
            "retrieved_by"
        ]:

            fused[key][
                "retrieved_by"
            ].append(
                "Dense"
            )

    ranked = sorted(
        fused.values(),
        key=lambda item:
            item["rrf_score"],
        reverse=True
    )

    return ranked[
        :top_k
    ]


# ============================================================
# Metrics
# ============================================================

def precision_at_k(
    retrieved_keys,
    relevant_keys,
    k
):
    retrieved = retrieved_keys[:k]

    if k <= 0:
        return 0.0

    hits = sum(
        1
        for key in retrieved
        if key in relevant_keys
    )

    return (
        hits
        /
        k
    )


def recall_at_k(
    retrieved_keys,
    relevant_keys,
    k
):
    if not relevant_keys:
        return 0.0

    retrieved = retrieved_keys[:k]

    hits = len(
        set(retrieved)
        &
        relevant_keys
    )

    return (
        hits
        /
        len(relevant_keys)
    )


def reciprocal_rank(
    retrieved_keys,
    relevant_keys
):
    for rank, key in enumerate(
        retrieved_keys,
        start=1
    ):

        if key in relevant_keys:

            return (
                1.0
                /
                rank
            )

    return 0.0


def ndcg_at_k(
    retrieved_keys,
    relevant_keys,
    k
):
    if not relevant_keys:
        return 0.0

    retrieved = retrieved_keys[:k]

    dcg = 0.0

    for rank, key in enumerate(
        retrieved,
        start=1
    ):

        if key in relevant_keys:

            dcg += (
                1.0
                /
                math.log2(
                    rank + 1
                )
            )

    ideal_hits = min(
        len(relevant_keys),
        k
    )

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1.0
        /
        math.log2(
            rank + 1
        )
        for rank in range(
            1,
            ideal_hits + 1
        )
    )

    if idcg == 0:
        return 0.0

    return (
        dcg
        /
        idcg
    )


def calculate_metrics(
    results,
    relevant_keys,
    k
):
    retrieved_keys = []

    for item in results:

        key = result_document_key(
            item
        )

        if key:
            retrieved_keys.append(
                key
            )

    return {
        "precision_at_10":
            precision_at_k(
                retrieved_keys,
                relevant_keys,
                k
            ),

        "recall_at_10":
            recall_at_k(
                retrieved_keys,
                relevant_keys,
                k
            ),

        "mrr":
            reciprocal_rank(
                retrieved_keys,
                relevant_keys
            ),

        "ndcg_at_10":
            ndcg_at_k(
                retrieved_keys,
                relevant_keys,
                k
            )
    }


# ============================================================
# Adaptive fusion using EXISTING retrieval results
# ============================================================

def adaptive_fusion_from_results(
    query,
    dense_docs,
    dense_distances,
    dense_ids,
    dense_metadatas,
    bm25_docs,
    bm25_scores,
    bm25_ids,
    bm25_metadatas,
    top_k
):
    """
    Predict adaptive weights from already-retrieved BM25/Dense
    results and fuse them.

    This avoids running BM25/Dense for a second time.
    """

    feature_start = (
        time.perf_counter()
    )

    retrieval_features = (
        retrieval_feature_service.extract(

            bm25_scores=bm25_scores,

            dense_distances=dense_distances,

            bm25_docs=bm25_docs,

            dense_docs=dense_docs

        )
    )

    prediction = (
        adaptive_predictor_v3.predict(

            query=query,

            retrieval_features=
                retrieval_features

        )
    )

    bm25_weight = float(
        prediction[
            "bm25_weight"
        ]
    )

    dense_weight = float(
        prediction[
            "dense_weight"
        ]
    )

    fused_results = (
        fusion_service.fuse(

            dense_docs=
                dense_docs,

            dense_distances=
                dense_distances,

            dense_ids=
                dense_ids,

            dense_metadatas=
                dense_metadatas,

            bm25_docs=
                bm25_docs,

            bm25_scores=
                bm25_scores,

            bm25_ids=
                bm25_ids,

            bm25_metadatas=
                bm25_metadatas,

            dense_weight=
                dense_weight,

            bm25_weight=
                bm25_weight

        )
    )

    feature_fusion_latency = (
        time.perf_counter()
        -
        feature_start
    )

    return (
        fused_results[
            :top_k
        ],

        bm25_weight,

        dense_weight,

        feature_fusion_latency
    )


# ============================================================
# Evaluate one query
# ============================================================

def evaluate_query(
    query_record,
    top_k,
    candidate_k=CANDIDATE_K
):
    query = query_record.get(
        "query",
        ""
    )

    query_id = query_record.get(
        "query_id",
        ""
    )

    dataset = query_record.get(
        "dataset",
        "unknown"
    )

    domain = query_record.get(
        "domain",
        "unknown"
    )

    # --------------------------------------------------------
    # Relevant documents
    # --------------------------------------------------------

    relevant_keys = set()

    for item in query_record.get(
        "relevant_documents",
        []
    ):

        relevance = int(
            item.get(
                "relevance",
                0
            )
        )

        if relevance <= 0:
            continue

        document_id = item.get(
            "document_id"
        )

        if document_id is None:
            continue

        relevant_keys.add(
            qrel_document_key(
                dataset,
                document_id
            )
        )

    # --------------------------------------------------------
    # ONE BM25 retrieval
    # --------------------------------------------------------

    bm25_start = (
        time.perf_counter()
    )

    (
        bm25_docs,
        bm25_scores,
        bm25_ids,
        bm25_metadatas
    ) = bm25_search(
        query,
        candidate_k
    )

    bm25_latency = (
        time.perf_counter()
        -
        bm25_start
    )

    bm25_results = make_results(
        bm25_docs,
        bm25_ids,
        bm25_metadatas,
        bm25_scores
    )

    # --------------------------------------------------------
    # ONE Dense retrieval
    # --------------------------------------------------------

    dense_start = (
        time.perf_counter()
    )

    (
        dense_docs,
        dense_distances,
        dense_ids,
        dense_metadatas
    ) = dense_search(
        query,
        candidate_k
    )

    dense_latency = (
        time.perf_counter()
        -
        dense_start
    )

    dense_results = make_results(
        dense_docs,
        dense_ids,
        dense_metadatas,
        dense_distances
    )

    rows = []

    # ========================================================
    # 1. BM25
    # ========================================================

    metrics = calculate_metrics(
        bm25_results,
        relevant_keys,
        top_k
    )

    rows.append(
        {
            "query_id":
                query_id,

            "domain":
                domain,

            "dataset":
                dataset,

            "method":
                "BM25",

            **metrics,

            "latency":
                bm25_latency,

            "bm25_weight":
                1.0,

            "dense_weight":
                0.0
        }
    )

    # ========================================================
    # 2. Dense
    # ========================================================

    metrics = calculate_metrics(
        dense_results,
        relevant_keys,
        top_k
    )

    rows.append(
        {
            "query_id":
                query_id,

            "domain":
                domain,

            "dataset":
                dataset,

            "method":
                "Dense",

            **metrics,

            "latency":
                dense_latency,

            "bm25_weight":
                0.0,

            "dense_weight":
                1.0
        }
    )

    # ========================================================
    # 3. RRF
    # ========================================================

    rrf_start = (
        time.perf_counter()
    )

    rrf_results = (
        reciprocal_rank_fusion(

            bm25_results,
            dense_results,
            top_k

        )
    )

    rrf_latency = (
        time.perf_counter()
        -
        rrf_start
    )

    metrics = calculate_metrics(
        rrf_results,
        relevant_keys,
        top_k
    )

    rows.append(
        {
            "query_id":
                query_id,

            "domain":
                domain,

            "dataset":
                dataset,

            "method":
                "RRF",

            **metrics,

            "latency":
                rrf_latency,

            "bm25_weight":
                None,

            "dense_weight":
                None
        }
    )

    # ========================================================
    # 4. Fixed Hybrid
    # ========================================================

    fixed_start = (
        time.perf_counter()
    )

    fixed_results = (
        fusion_service.fuse(

            dense_docs=
                dense_docs,

            dense_distances=
                dense_distances,

            dense_ids=
                dense_ids,

            dense_metadatas=
                dense_metadatas,

            bm25_docs=
                bm25_docs,

            bm25_scores=
                bm25_scores,

            bm25_ids=
                bm25_ids,

            bm25_metadatas=
                bm25_metadatas,

            dense_weight=
                FIXED_DENSE_WEIGHT,

            bm25_weight=
                FIXED_BM25_WEIGHT

        )[:top_k]
    )

    fixed_latency = (
        time.perf_counter()
        -
        fixed_start
    )

    metrics = calculate_metrics(
        fixed_results,
        relevant_keys,
        top_k
    )

    rows.append(
        {
            "query_id":
                query_id,

            "domain":
                domain,

            "dataset":
                dataset,

            "method":
                "Fixed Hybrid",

            **metrics,

            "latency":
                fixed_latency,

            "bm25_weight":
                FIXED_BM25_WEIGHT,

            "dense_weight":
                FIXED_DENSE_WEIGHT
        }
    )

    # ========================================================
    # 5. Adaptive Hybrid
    # ========================================================

    (
        adaptive_results,
        adaptive_bm25_weight,
        adaptive_dense_weight,
        adaptive_fusion_latency
    ) = adaptive_fusion_from_results(

        query,

        dense_docs,
        dense_distances,
        dense_ids,
        dense_metadatas,

        bm25_docs,
        bm25_scores,
        bm25_ids,
        bm25_metadatas,

        top_k
    )

    metrics = calculate_metrics(
        adaptive_results,
        relevant_keys,
        top_k
    )

    rows.append(
        {
            "query_id":
                query_id,

            "domain":
                domain,

            "dataset":
                dataset,

            "method":
                "Adaptive Hybrid",

            **metrics,

            "latency":
                adaptive_fusion_latency,

            "bm25_weight":
                adaptive_bm25_weight,

            "dense_weight":
                adaptive_dense_weight
        }
    )

    return rows


# ============================================================
# Overall Summary
# ============================================================

def build_summary(
    rows
):
    grouped = defaultdict(list)

    for row in rows:

        grouped[
            row["method"]
        ].append(
            row
        )

    summary = []

    for method in METHOD_ORDER:

        method_rows = grouped.get(
            method,
            []
        )

        if not method_rows:
            continue

        summary.append(
            {
                "method":
                    method,

                "queries":
                    len(method_rows),

                "precision_at_10":
                    sum(
                        float(
                            row[
                                "precision_at_10"
                            ]
                        )
                        for row in method_rows
                    )
                    /
                    len(method_rows),

                "recall_at_10":
                    sum(
                        float(
                            row[
                                "recall_at_10"
                            ]
                        )
                        for row in method_rows
                    )
                    /
                    len(method_rows),

                "mrr":
                    sum(
                        float(
                            row["mrr"]
                        )
                        for row in method_rows
                    )
                    /
                    len(method_rows),

                "ndcg_at_10":
                    sum(
                        float(
                            row[
                                "ndcg_at_10"
                            ]
                        )
                        for row in method_rows
                    )
                    /
                    len(method_rows),

                "latency":
                    sum(
                        float(
                            row["latency"]
                        )
                        for row in method_rows
                    )
                    /
                    len(method_rows)
            }
        )

    return summary


# ============================================================
# Domain Summary
# ============================================================

def build_domain_summary(
    rows
):
    grouped = defaultdict(list)

    for row in rows:

        grouped[
            (
                row["domain"],
                row["method"]
            )
        ].append(
            row
        )

    output = []

    for (
        domain,
        method
    ), domain_rows in sorted(
        grouped.items()
    ):

        output.append(
            {
                "domain":
                    domain,

                "method":
                    method,

                "queries":
                    len(domain_rows),

                "precision_at_10":
                    sum(
                        float(
                            row[
                                "precision_at_10"
                            ]
                        )
                        for row in domain_rows
                    )
                    /
                    len(domain_rows),

                "recall_at_10":
                    sum(
                        float(
                            row[
                                "recall_at_10"
                            ]
                        )
                        for row in domain_rows
                    )
                    /
                    len(domain_rows),

                "mrr":
                    sum(
                        float(
                            row["mrr"]
                        )
                        for row in domain_rows
                    )
                    /
                    len(domain_rows),

                "ndcg_at_10":
                    sum(
                        float(
                            row[
                                "ndcg_at_10"
                            ]
                        )
                        for row in domain_rows
                    )
                    /
                    len(domain_rows),

                "latency":
                    sum(
                        float(
                            row["latency"]
                        )
                        for row in domain_rows
                    )
                    /
                    len(domain_rows)
            }
        )

    return output


# ============================================================
# Weight Summary
# ============================================================

def build_weight_summary(
    rows
):
    adaptive_rows = [
        row
        for row in rows
        if row["method"]
        == "Adaptive Hybrid"
    ]

    if not adaptive_rows:
        return []

    grouped = defaultdict(list)

    for row in adaptive_rows:

        grouped[
            row["domain"]
        ].append(
            row
        )

    output = []

    for domain, domain_rows in sorted(
        grouped.items()
    ):

        bm25_values = [
            float(
                row["bm25_weight"]
            )
            for row in domain_rows
            if row["bm25_weight"]
            is not None
        ]

        dense_values = [
            float(
                row["dense_weight"]
            )
            for row in domain_rows
            if row["dense_weight"]
            is not None
        ]

        if not bm25_values:
            continue

        output.append(
            {
                "domain":
                    domain,

                "queries":
                    len(domain_rows),

                "avg_bm25_weight":
                    sum(
                        bm25_values
                    )
                    /
                    len(
                        bm25_values
                    ),

                "min_bm25_weight":
                    min(
                        bm25_values
                    ),

                "max_bm25_weight":
                    max(
                        bm25_values
                    ),

                "avg_dense_weight":
                    (
                        sum(
                            dense_values
                        )
                        /
                        len(
                            dense_values
                        )
                        if dense_values
                        else 0.0
                    )
            }
        )

    return output


# ============================================================
# CSV
# ============================================================

def write_csv(
    path,
    rows,
    fieldnames
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# Console
# ============================================================

def print_summary(
    summary
):
    print()

    print(
        "## Method"
        "                    "
        "P@10      "
        "Recall@10      "
        "MRR       "
        "nDCG@10      "
        "Latency"
    )

    print(
        "-" * 90
    )

    for row in summary:

        print(
            f"{row['method']:<24}"
            f"{row['precision_at_10']:>8.4f}"
            f"{row['recall_at_10']:>13.4f}"
            f"{row['mrr']:>11.4f}"
            f"{row['ndcg_at_10']:>13.4f}"
            f"{row['latency']:>10.4f}"
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Balanced multi-domain "
            "Adaptive Hybrid RAG benchmark."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Number of benchmark queries "
            "to evaluate."
        )
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate benchmark
    # --------------------------------------------------------

    if not BENCHMARK_FILE.exists():

        raise FileNotFoundError(
            "Benchmark file not found:\n"
            f"{BENCHMARK_FILE}"
        )

    queries = load_jsonl(
        BENCHMARK_FILE
    )

    if args.limit is not None:

        if args.limit <= 0:

            raise ValueError(
                "--limit must be > 0."
            )

        queries = queries[
            :args.limit
        ]

    print()
    print(
        "=" * 80
    )
    print(
        "MULTI-DOMAIN ADAPTIVE HYBRID RAG BENCHMARK"
    )
    print(
        "=" * 80
    )

    print(
        f"Benchmark queries : "
        f"{len(queries)}"
    )

    print(
        f"Top-K             : "
        f"{args.top_k}"
    )

    print(
        f"Candidate-K       : "
        f"{CANDIDATE_K}"
    )

    print(
        "Methods            : "
        "BM25, Dense, RRF, "
        "Fixed Hybrid, Adaptive Hybrid"
    )

    print(
        "=" * 80
    )

    all_rows = []

    benchmark_start = (
        time.perf_counter()
    )

    total = len(
        queries
    )

    for index, query_record in enumerate(
        queries,
        start=1
    ):

        query_start = (
            time.perf_counter()
        )

        rows = evaluate_query(
            query_record,
            args.top_k,
            CANDIDATE_K
        )

        all_rows.extend(
            rows
        )

        query_elapsed = (
            time.perf_counter()
            -
            query_start
        )

        total_elapsed = (
            time.perf_counter()
            -
            benchmark_start
        )

        rate = (
            index
            /
            total_elapsed
        )

        print(
            f"Processed "
            f"{index:>4}/{total}"
            f" | "
            f"{query_elapsed:>7.2f}s/query"
            f" | "
            f"{rate:>6.2f} queries/sec"
        )

    # --------------------------------------------------------
    # Build outputs
    # --------------------------------------------------------

    summary = build_summary(
        all_rows
    )

    domain_summary = (
        build_domain_summary(
            all_rows
        )
    )

    weight_summary = (
        build_weight_summary(
            all_rows
        )
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    detailed_path = (
        RESULTS_DIR
        / "multidomain_query_results.csv"
    )

    summary_path = (
        RESULTS_DIR
        / "multidomain_summary.csv"
    )

    domain_path = (
        RESULTS_DIR
        / "multidomain_domain_summary.csv"
    )

    weight_path = (
        RESULTS_DIR
        / "adaptive_weight_summary.csv"
    )

    detailed_fields = [
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
        "dense_weight"
    ]

    summary_fields = [
        "method",
        "queries",
        "precision_at_10",
        "recall_at_10",
        "mrr",
        "ndcg_at_10",
        "latency"
    ]

    domain_fields = [
        "domain",
        "method",
        "queries",
        "precision_at_10",
        "recall_at_10",
        "mrr",
        "ndcg_at_10",
        "latency"
    ]

    weight_fields = [
        "domain",
        "queries",
        "avg_bm25_weight",
        "min_bm25_weight",
        "max_bm25_weight",
        "avg_dense_weight"
    ]

    write_csv(
        detailed_path,
        all_rows,
        detailed_fields
    )

    write_csv(
        summary_path,
        summary,
        summary_fields
    )

    write_csv(
        domain_path,
        domain_summary,
        domain_fields
    )

    write_csv(
        weight_path,
        weight_summary,
        weight_fields
    )

    print_summary(
        summary
    )

    print()
    print(
        "Detailed : "
        f"{detailed_path}"
    )

    print(
        "Summary  : "
        f"{summary_path}"
    )

    print(
        "Domains  : "
        f"{domain_path}"
    )

    print(
        "Weights  : "
        f"{weight_path}"
    )

    print()
    print(
        "=" * 80
    )
    print(
        "BENCHMARK COMPLETE"
    )
    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()
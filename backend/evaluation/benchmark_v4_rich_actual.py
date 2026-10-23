from pathlib import Path
import sys

# Add backend directory to Python path
BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import csv
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from collections import defaultdict

from services.bm25_service import bm25_search
from services.retrieval_service import dense_search
from services.adaptive_predictor_v4 import adaptive_predictor_v4
from services.retrieval_feature_service import (
    V4_RICH_FEATURES,
    V4_RICH_FEATURE_TOP_K,
    retrieval_feature_service,
)
from services.fusion_service import fusion_service
from config.settings import COLLECTION_NAME, EMBEDDING_MODEL

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
    / "corrected_v4_audit"
    / "runs"
)
BENCHMARK_MANIFEST = (
    BASE_DIR
    / "evaluation"
    / "dataset_v2"
    / "benchmark"
    / "benchmark_manifest.json"
)
ADAPTIVE_MODEL_FILE = (
    BASE_DIR
    / "models"
    / "adaptive_weight_model_v4_rich.pkl"
)


# ============================================================
# Configuration
# ============================================================

TOP_K = 10
CANDIDATE_K = 50

FIXED_BM25_WEIGHT = 0.50
FIXED_DENSE_WEIGHT = 0.50
RRF_K = 60


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


def create_run_directory():
    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    dense_metadatas,
    bm25_metadatas,
    k=60,
):

    if len(dense_ids) != len(dense_metadatas):
        raise ValueError(
            "dense_ids and dense_metadatas must have equal lengths."
        )
    if len(bm25_ids) != len(bm25_metadatas):
        raise ValueError(
            "bm25_ids and bm25_metadatas must have equal lengths."
        )

    scores = defaultdict(float)

    metadata = {}

    for rank, (document_id, item_metadata) in enumerate(
        zip(dense_ids, dense_metadatas),
        start=1,
    ):

        document_id = str(
            document_id
        )

        scores[document_id] += (
            1.0
            / (k + rank)
        )
        metadata.setdefault(document_id, item_metadata or {})

    for rank, (document_id, item_metadata) in enumerate(
        zip(bm25_ids, bm25_metadatas),
        start=1,
    ):

        document_id = str(
            document_id
        )

        scores[document_id] += (
            1.0
            / (k + rank)
        )
        metadata.setdefault(document_id, item_metadata or {})

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
    run_dir = create_run_directory()
    manifest = json.loads(
        BENCHMARK_MANIFEST.read_text(encoding="utf-8")
    )
    expected_query_count = manifest["total_queries"]
    if len(queries) != expected_query_count:
        raise ValueError(
            f"Expected {expected_query_count} benchmark queries, "
            f"found {len(queries)}."
        )
    query_ids = [str(row["query_id"]) for row in queries]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("Benchmark query IDs must be unique.")

    run_manifest = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_file": str(BENCHMARK_FILE.relative_to(BASE_DIR)),
        "benchmark_version": manifest["benchmark_version"],
        "dataset_names": sorted(manifest["datasets"]),
        "queries_by_dataset": {
            dataset: sum(
                row.get("dataset") == dataset
                for row in queries
            )
            for dataset in sorted(manifest["datasets"])
        },
        "random_seed": manifest.get("random_seed"),
        "benchmark_manifest_file": str(
            BENCHMARK_MANIFEST.relative_to(BASE_DIR)
        ),
        "benchmark_manifest_sha256": sha256_file(BENCHMARK_MANIFEST),
        "qrels_source": "relevant_documents embedded in benchmark_queries.jsonl",
        "benchmark_queries_sha256": sha256_file(BENCHMARK_FILE),
        "query_count": len(queries),
        "candidate_depth": CANDIDATE_K,
        "feature_top_k": V4_RICH_FEATURE_TOP_K,
        "final_top_k": TOP_K,
        "feature_schema": list(V4_RICH_FEATURES),
        "adaptive_model": {
            "path": str(ADAPTIVE_MODEL_FILE.relative_to(BASE_DIR)),
            "sha256": sha256_file(ADAPTIVE_MODEL_FILE),
        },
        "retrieval_config": {
            "collection": COLLECTION_NAME,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_normalized": False,
            "dense_distance_space": "l2",
        },
        "methods": [
            "BM25",
            "Dense",
            "Fixed Hybrid (0.50)",
            "Adaptive Hybrid",
            "RRF",
        ],
        "latency_components": [
            "BM25 retrieval",
            "dense retrieval",
            "feature extraction",
            "adaptive prediction",
            "fusion",
        ],
        "reranking_executed": False,
        "generation_executed": False,
        "training_evaluation_independence": "unverified",
        "evaluation_set_derived_weights_used": False,
        "latency_methodology": (
            "per-query wall time for retrieval-only (applicable BM25 and/or "
            "dense retrieval) and complete retrieval pipeline (retrieval, "
            "feature extraction, prediction, fusion); reranking and "
            "generation are excluded"
        ),
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2),
        encoding="utf-8",
    )

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
        "RRF",
        "Fixed Hybrid (0.50)",
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
    query_method_rows = []
    timing_rows = []

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

        feature_start = time.perf_counter()
        features = (
            retrieval_feature_service.extract(
                query=query,
                bm25_scores=bm25_scores,
                dense_distances=dense_distances,
                bm25_docs=bm25_docs,
                dense_docs=dense_docs,
                bm25_ids=bm25_ids,
                dense_ids=dense_ids,
            )
        )
        feature_latency = time.perf_counter() - feature_start

        prediction_start = time.perf_counter()
        prediction = (
            adaptive_predictor_v4.predict(
                query=query,
                retrieval_features=features,
            )
        )
        prediction_latency = time.perf_counter() - prediction_start

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

        fixed_start = time.perf_counter()
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
        fixed_latency = time.perf_counter() - fixed_start

        fixed_results = normalize_results(
            fixed_fused
        )

        rrf_start = time.perf_counter()
        rrf_results = reciprocal_rank_fusion(
            dense_ids,
            bm25_ids,
            dense_metadatas,
            bm25_metadatas,
            k=RRF_K,
        )
        rrf_latency = time.perf_counter() - rrf_start

        # ====================================================
        # Adaptive V4 Hybrid
        # ====================================================

        adaptive_start = time.perf_counter()
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
        adaptive_latency = time.perf_counter() - adaptive_start

        adaptive_results = normalize_results(
            adaptive_fused
        )

        # ====================================================
        # Metrics
        # ====================================================

        result_sets = {
            "BM25": bm25_results,
            "Dense": dense_results,
            "RRF": rrf_results,
            "Fixed Hybrid (0.50)": fixed_results,
            "Adaptive Hybrid": adaptive_results,
        }

        query_metrics = {}

        for method, results in result_sets.items():

            if method == "BM25":
                components = {
                    "bm25_retrieval_seconds": bm25_latency,
                    "dense_retrieval_seconds": 0.0,
                    "feature_extraction_seconds": 0.0,
                    "adaptive_prediction_seconds": 0.0,
                    "fusion_seconds": 0.0,
                }
            elif method == "Dense":
                components = {
                    "bm25_retrieval_seconds": 0.0,
                    "dense_retrieval_seconds": dense_latency,
                    "feature_extraction_seconds": 0.0,
                    "adaptive_prediction_seconds": 0.0,
                    "fusion_seconds": 0.0,
                }
            else:
                components = {
                    "bm25_retrieval_seconds": bm25_latency,
                    "dense_retrieval_seconds": dense_latency,
                    "feature_extraction_seconds": 0.0,
                    "adaptive_prediction_seconds": 0.0,
                    "fusion_seconds": 0.0,
                }
                if method == "RRF":
                    components["fusion_seconds"] = rrf_latency
                elif method == "Fixed Hybrid (0.50)":
                    components["fusion_seconds"] = fixed_latency
                else:
                    components["feature_extraction_seconds"] = feature_latency
                    components["adaptive_prediction_seconds"] = prediction_latency
                    components["fusion_seconds"] = adaptive_latency

            latency = sum(components.values())

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

            query_method_rows.append(
                {
                    "query_id": row["query_id"],
                    "domain": domain,
                    "method": method,
                    "precision_at_10": p,
                    "recall_at_10": r,
                    "mrr": m,
                    "ndcg_at_10": n,
                    "latency_seconds": latency,
                }
            )
            timing_rows.append(
                {
                    "query_id": row["query_id"],
                    "domain": domain,
                    "method": method,
                    **components,
                    "total_pipeline_seconds": latency,
                    "reranking_seconds": 0.0,
                    "generation_seconds": 0.0,
                }
            )

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
                        "Fixed Hybrid (0.50)"
                    ]["mrr"],

                "adaptive_mrr":
                    query_metrics[
                        "Adaptive Hybrid"
                    ]["mrr"],

                "fixed_ndcg":
                    query_metrics[
                        "Fixed Hybrid (0.50)"
                    ]["ndcg"],

                "adaptive_ndcg":
                    query_metrics[
                        "Adaptive Hybrid"
                    ]["ndcg"],

                "fixed_precision":
                    query_metrics[
                        "Fixed Hybrid (0.50)"
                    ]["precision"],

                "adaptive_precision":
                    query_metrics[
                        "Adaptive Hybrid"
                    ]["precision"],

                "fixed_recall":
                    query_metrics[
                        "Fixed Hybrid (0.50)"
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
        run_dir
        / "corrected_v4_summary.csv"
    )
    component_names = (
        "bm25_retrieval_seconds",
        "dense_retrieval_seconds",
        "feature_extraction_seconds",
        "adaptive_prediction_seconds",
        "fusion_seconds",
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
                "mean_bm25_retrieval_seconds",
                "mean_dense_retrieval_seconds",
                "mean_feature_extraction_seconds",
                "mean_adaptive_prediction_seconds",
                "mean_fusion_seconds",
                "latency_seconds",
            ]
        )

        for method in methods:

            data = totals[method]
            count = data["queries"]
            method_timings = [
                timing
                for timing in timing_rows
                if timing["method"] == method
            ]

            writer.writerow(
                [
                    method,
                    count,
                    data["precision"] / count,
                    data["recall"] / count,
                    data["mrr"] / count,
                    data["ndcg"] / count,
                    *[
                        sum(item[name] for item in method_timings) / count
                        for name in component_names
                    ],
                    data["latency"] / count,
                ]
            )

    # ========================================================
    # Domain Summary
    # ========================================================

    domain_path = (
        run_dir
        / "corrected_v4_domain_summary.csv"
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
                "mean_bm25_retrieval_seconds",
                "mean_dense_retrieval_seconds",
                "mean_feature_extraction_seconds",
                "mean_adaptive_prediction_seconds",
                "mean_fusion_seconds",
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
                domain_method_timings = [
                    timing
                    for timing in timing_rows
                    if timing["method"] == method
                    and timing["domain"] == domain
                ]

                writer.writerow(
                    [
                        domain,
                        method,
                        count,
                        data["precision"] / count,
                        data["recall"] / count,
                        data["mrr"] / count,
                        data["ndcg"] / count,
                        *[
                            sum(item[name] for item in domain_method_timings)
                            / count
                            for name in component_names
                        ],
                        data["latency"] / count,
                    ]
                )

    # ========================================================
    # Query Comparison
    # ========================================================

    comparison_path = (
        run_dir
        / "corrected_v4_adaptive_vs_fixed_query_comparison.csv"
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
        run_dir
        / "corrected_v4_adaptive_weights.csv"
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

    query_methods_path = run_dir / "corrected_v4_query_methods.csv"
    with open(
        query_methods_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=query_method_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(query_method_rows)

    timings_path = run_dir / "corrected_v4_stage_timings.csv"
    with open(
        timings_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=timing_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(timing_rows)

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
        run_dir
        / "corrected_v4_improvement_vs_fixed_050.csv"
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
    run_manifest["status"] = "completed"
    run_manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    run_manifest["outputs"] = [
        str(path.name)
        for path in (
            summary_path,
            domain_path,
            comparison_path,
            weights_path,
            query_methods_path,
            timings_path,
            improvement_path,
        )
    ]
    (run_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Run output: {run_dir}")

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
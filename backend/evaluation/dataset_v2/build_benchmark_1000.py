from pathlib import Path
import json
import random


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    BASE_DIR
    / "evaluation"
    / "dataset_v2"
    / "processed"
)

BENCHMARK_DIR = (
    BASE_DIR
    / "evaluation"
    / "dataset_v2"
    / "benchmark"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

QUERIES_PER_DOMAIN = 250

TOTAL_QUERIES = 1000


# ============================================================
# DATASETS
# ============================================================

DATASETS = {

    "scifact": {
        "domain": "scientific"
    },

    "fiqa": {
        "domain": "finance"
    },

    "nfcorpus": {
        "domain": "medical"
    },

    "arguana": {
        "domain": "argumentation"
    }

}


# ============================================================
# HELPERS
# ============================================================

def load_jsonl(path):
    """
    Load a JSONL file.
    """

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


def save_jsonl(path, rows):
    """
    Save rows as JSONL.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        for row in rows:

            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
                + "\n"
            )


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(name, config):
    """
    Load queries and test qrels for one dataset.

    Only queries having at least one relevant document
    are eligible for the benchmark.
    """

    dataset_dir = (
        PROCESSED_DIR
        / name
    )

    queries_path = (
        dataset_dir
        / "queries.jsonl"
    )

    qrels_path = (
        dataset_dir
        / "qrels_test.jsonl"
    )

    print()
    print("=" * 70)
    print(
        f"Loading dataset: {name}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Queries
    # --------------------------------------------------------

    queries = load_jsonl(
        queries_path
    )

    print(
        f"Queries loaded : {len(queries)}"
    )

    # --------------------------------------------------------
    # Qrels
    # --------------------------------------------------------

    qrels = load_jsonl(
        qrels_path
    )

    print(
        f"Qrels loaded   : {len(qrels)}"
    )

    # --------------------------------------------------------
    # Build relevance mapping
    # --------------------------------------------------------

    relevant_documents = {}

    for row in qrels:

        query_id = str(
            row["query_id"]
        )

        document_id = str(
            row["document_id"]
        )

        relevance = int(
            row.get(
                "relevance",
                0
            )
        )

        if relevance <= 0:
            continue

        if query_id not in relevant_documents:

            relevant_documents[
                query_id
            ] = []

        relevant_documents[
            query_id
        ].append(
            {
                "document_id":
                    document_id,

                "relevance":
                    relevance
            }
        )

    # --------------------------------------------------------
    # Keep only valid queries
    # --------------------------------------------------------

    valid_queries = []

    for query in queries:

        source_query_id = str(
            query["query_id"]
        )

        if source_query_id not in relevant_documents:
            continue

        query_text = (
            query.get("query")
            or query.get("text")
            or ""
        ).strip()

        if not query_text:
            continue

        valid_queries.append(
            {
                "source_query_id":
                    source_query_id,

                "query":
                    query_text,

                "domain":
                    config["domain"],

                "dataset":
                    name,

                "source":
                    "BEIR",

                "relevant_documents":
                    relevant_documents[
                        source_query_id
                    ]
            }
        )

    print(
        f"Valid queries  : {len(valid_queries)}"
    )

    return valid_queries


# ============================================================
# CREATE GLOBALLY UNIQUE QUERY ID
# ============================================================

def make_global_query_id(
    dataset_name,
    source_query_id
):
    """
    Create a globally unique query identifier.

    Native BEIR query IDs are only unique inside their
    respective datasets. Prefixing the dataset prevents
    collisions when multiple datasets are combined.
    """

    return (
        f"{dataset_name}-"
        f"{source_query_id}"
    )


# ============================================================
# BUILD BENCHMARK
# ============================================================

def build_benchmark():

    print()
    print("=" * 80)
    print(
        "BALANCED MULTI-DOMAIN BENCHMARK BUILDER"
    )
    print("=" * 80)

    print()
    print(
        f"Random seed : {RANDOM_SEED}"
    )

    print(
        f"Target      : "
        f"{QUERIES_PER_DOMAIN} queries per domain"
    )

    print(
        f"Total       : "
        f"{TOTAL_QUERIES} queries"
    )

    # --------------------------------------------------------
    # Random generator
    # --------------------------------------------------------

    rng = random.Random(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Load all datasets
    # --------------------------------------------------------

    dataset_queries = {}

    for name, config in DATASETS.items():

        dataset_queries[name] = (
            load_dataset(
                name,
                config
            )
        )

    # --------------------------------------------------------
    # Validate availability
    # --------------------------------------------------------

    for name in DATASETS:

        available = len(
            dataset_queries[name]
        )

        if available < QUERIES_PER_DOMAIN:

            raise ValueError(
                f"Dataset '{name}' has only "
                f"{available} valid queries, "
                f"but {QUERIES_PER_DOMAIN} "
                f"are required."
            )

    # --------------------------------------------------------
    # Sample balanced queries
    # --------------------------------------------------------

    benchmark_rows = []

    for name in DATASETS:

        candidates = list(
            dataset_queries[name]
        )

        selected = rng.sample(
            candidates,
            QUERIES_PER_DOMAIN
        )

        for row in selected:

            source_query_id = (
                row["source_query_id"]
            )

            global_query_id = (
                make_global_query_id(
                    name,
                    source_query_id
                )
            )

            benchmark_rows.append(
                {
                    "query_id":
                        global_query_id,

                    "source_query_id":
                        source_query_id,

                    "query":
                        row["query"],

                    "domain":
                        row["domain"],

                    "dataset":
                        row["dataset"],

                    "source":
                        row["source"],

                    "relevant_documents":
                        row[
                            "relevant_documents"
                        ]
                }
            )

    # --------------------------------------------------------
    # Shuffle final benchmark
    # --------------------------------------------------------

    rng.shuffle(
        benchmark_rows
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    print()
    print("=" * 70)
    print(
        "BENCHMARK VALIDATION"
    )
    print("=" * 70)

    actual_queries = len(
        benchmark_rows
    )

    print(
        f"Expected queries : "
        f"{TOTAL_QUERIES}"
    )

    print(
        f"Actual queries   : "
        f"{actual_queries}"
    )

    if actual_queries != TOTAL_QUERIES:

        raise ValueError(
            "Benchmark query count mismatch."
        )

    # --------------------------------------------------------
    # Check global query ID uniqueness
    # --------------------------------------------------------

    query_ids = [
        row["query_id"]
        for row in benchmark_rows
    ]

    unique_query_ids = set(
        query_ids
    )

    print(
        f"Unique query IDs : "
        f"{len(unique_query_ids)}"
    )

    if len(unique_query_ids) != TOTAL_QUERIES:

        duplicates = []

        seen = set()

        for query_id in query_ids:

            if query_id in seen:

                duplicates.append(
                    query_id
                )

            seen.add(
                query_id
            )

        raise ValueError(
            "Global query ID collision detected: "
            f"{duplicates}"
        )

    # --------------------------------------------------------
    # Dataset distribution
    # --------------------------------------------------------

    print()
    print(
        "Queries by dataset:"
    )

    dataset_counts = {}

    for row in benchmark_rows:

        dataset = row["dataset"]

        dataset_counts[dataset] = (
            dataset_counts.get(
                dataset,
                0
            )
            + 1
        )

    for dataset in sorted(
        dataset_counts
    ):

        print(
            f"{dataset:<20}: "
            f"{dataset_counts[dataset]}"
        )

        if (
            dataset_counts[dataset]
            != QUERIES_PER_DOMAIN
        ):

            raise ValueError(
                f"Dataset balance error for "
                f"{dataset}."
            )

    # --------------------------------------------------------
    # Domain distribution
    # --------------------------------------------------------

    print()
    print(
        "Queries by domain:"
    )

    domain_counts = {}

    for row in benchmark_rows:

        domain = row["domain"]

        domain_counts[domain] = (
            domain_counts.get(
                domain,
                0
            )
            + 1
        )

    for domain in sorted(
        domain_counts
    ):

        print(
            f"{domain:<20}: "
            f"{domain_counts[domain]}"
        )

        if (
            domain_counts[domain]
            != QUERIES_PER_DOMAIN
        ):

            raise ValueError(
                f"Domain balance error for "
                f"{domain}."
            )

    # ========================================================
    # SAVE BENCHMARK
    # ========================================================

    BENCHMARK_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    benchmark_path = (
        BENCHMARK_DIR
        / "benchmark_queries.jsonl"
    )

    manifest_path = (
        BENCHMARK_DIR
        / "benchmark_manifest.json"
    )

    save_jsonl(
        benchmark_path,
        benchmark_rows
    )

    # --------------------------------------------------------
    # Manifest statistics
    # --------------------------------------------------------

    dataset_manifest = {}

    for dataset_name in DATASETS:

        rows = [
            row
            for row in benchmark_rows
            if row["dataset"]
            == dataset_name
        ]

        relevant_pairs = sum(
            len(
                row[
                    "relevant_documents"
                ]
            )
            for row in rows
        )

        dataset_manifest[
            dataset_name
        ] = {

            "domain":
                DATASETS[
                    dataset_name
                ]["domain"],

            "queries":
                len(rows),

            "relevant_query_document_pairs":
                relevant_pairs
        }

    manifest = {

        "benchmark_version":
            "multidomain_v2_1000",

        "description":
            (
                "Balanced multi-domain retrieval "
                "benchmark constructed from public "
                "BEIR datasets."
            ),

        "random_seed":
            RANDOM_SEED,

        "total_queries":
            TOTAL_QUERIES,

        "domains":
            len(DATASETS),

        "queries_per_domain":
            QUERIES_PER_DOMAIN,

        "global_query_id_scheme":
            "dataset-source_query_id",

        "datasets":
            dataset_manifest,

        "evaluation_scope":
            [
                "BM25",
                "Dense",
                "RRF",
                "Fixed Hybrid",
                "Adaptive Hybrid"
            ],

        "metrics":
            [
                "Precision@10",
                "Recall@10",
                "MRR",
                "nDCG@10",
                "Latency"
            ]
    }

    with open(
        manifest_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # SUCCESS
    # ========================================================

    print()
    print(
        "BENCHMARK VALIDATION PASSED."
    )

    print()
    print("=" * 80)
    print(
        "BENCHMARK CREATED SUCCESSFULLY"
    )
    print("=" * 80)

    print()
    print(
        f"Queries : {actual_queries}"
    )

    print(
        f"Unique IDs : {len(unique_query_ids)}"
    )

    print(
        f"Output  : {benchmark_path}"
    )

    print(
        f"Manifest: {manifest_path}"
    )

    print()
    print(
        "Query IDs are globally unique using:"
    )

    print(
        "dataset-source_query_id"
    )

    print()
    print(
        "Next phase:"
    )

    print(
        "Run the multi-domain retrieval benchmark."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    build_benchmark()
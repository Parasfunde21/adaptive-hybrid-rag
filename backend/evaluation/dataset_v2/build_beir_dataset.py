from pathlib import Path
import json

from datasets import load_dataset


BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    BASE_DIR
    / "evaluation"
    / "dataset_v2"
    / "processed"
)


DATASETS = {
    "scifact": {
        "domain": "scientific",
    },

    "fiqa": {
        "domain": "finance",
    },

    "nfcorpus": {
        "domain": "medical",
    },

    "arguana": {
        "domain": "argumentation",
    },
}


def save_jsonl(path, rows):

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


def load_beir_dataset(name):

    print()
    print("=" * 70)
    print(f"LOADING BEIR DATASET: {name}")
    print("=" * 70)

    print(
        "Downloading/loading from Hugging Face..."
    )

    corpus = load_dataset(
        f"BeIR/{name}",
        "corpus",
        split="corpus"
    )

    queries = load_dataset(
        f"BeIR/{name}",
        "queries",
        split="queries"
    )

    print(
        f"Corpus rows : {len(corpus)}"
    )

    print(
        f"Query rows  : {len(queries)}"
    )

    return corpus, queries


def load_qrels(name):

    print()
    print(
        f"Loading qrels for {name}..."
    )

    try:

        qrels = load_dataset(
            f"BeIR/{name}-qrels",
            split="test"
        )

    except Exception as error:

        print(
            f"[WARNING] Could not load "
            f"BeIR/{name}-qrels"
        )

        print(
            f"Reason: {error}"
        )

        return None

    print(
        f"Qrels rows : {len(qrels)}"
    )

    return qrels


def process_dataset(
    name,
    config
):

    corpus, queries = (
        load_beir_dataset(name)
    )

    qrels = load_qrels(name)

    output_dir = (
        PROCESSED_DIR
        / name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ======================================================
    # CORPUS
    # ======================================================

    corpus_rows = []

    for row in corpus:

        document_id = str(
            row["_id"]
        )

        title = (
            row.get("title")
            or ""
        )

        text = (
            row.get("text")
            or ""
        )

        corpus_rows.append(
            {
                "document_id":
                    document_id,

                "title":
                    title,

                "text":
                    text,

                "domain":
                    config["domain"],

                "dataset":
                    name,

                "source":
                    "BEIR",
            }
        )

    corpus_path = (
        output_dir
        / "corpus.jsonl"
    )

    save_jsonl(
        corpus_path,
        corpus_rows
    )

    # ======================================================
    # QUERIES
    # ======================================================

    query_rows = []

    for row in queries:

        query_id = str(
            row["_id"]
        )

        query_text = (
            row.get("text")
            or ""
        )

        query_rows.append(
            {
                "query_id":
                    query_id,

                "query":
                    query_text,

                "domain":
                    config["domain"],

                "dataset":
                    name,

                "source":
                    "BEIR",
            }
        )

    queries_path = (
        output_dir
        / "queries.jsonl"
    )

    save_jsonl(
        queries_path,
        query_rows
    )

    # ======================================================
    # QRELS
    # ======================================================

    qrels_rows = []

    if qrels is not None:

        for row in qrels:

            query_id = str(
                row["query-id"]
            )

            document_id = str(
                row["corpus-id"]
            )

            score = int(
                row["score"]
            )

            qrels_rows.append(
                {
                    "query_id":
                        query_id,

                    "document_id":
                        document_id,

                    "relevance":
                        score,

                    "domain":
                        config["domain"],

                    "dataset":
                        name,

                    "source":
                        "BEIR",
                }
            )

    qrels_path = (
        output_dir
        / "qrels_test.jsonl"
    )

    save_jsonl(
        qrels_path,
        qrels_rows
    )

    # ======================================================
    # STATISTICS
    # ======================================================

    unique_relevant_queries = len(
        {
            row["query_id"]
            for row in qrels_rows
        }
    )

    print()
    print(
        "-" * 70
    )

    print(
        f"Dataset           : {name}"
    )

    print(
        f"Domain            : "
        f"{config['domain']}"
    )

    print(
        f"Corpus documents  : "
        f"{len(corpus_rows)}"
    )

    print(
        f"Queries           : "
        f"{len(query_rows)}"
    )

    print(
        f"Qrels             : "
        f"{len(qrels_rows)}"
    )

    print(
        f"Queries with qrels: "
        f"{unique_relevant_queries}"
    )

    print(
        f"Output            : "
        f"{output_dir}"
    )

    print(
        "-" * 70
    )


def create_manifest():

    datasets = {}

    total_documents = 0
    total_queries = 0
    total_qrels = 0

    for name, config in DATASETS.items():

        output_dir = (
            PROCESSED_DIR
            / name
        )

        corpus_path = (
            output_dir
            / "corpus.jsonl"
        )

        queries_path = (
            output_dir
            / "queries.jsonl"
        )

        qrels_path = (
            output_dir
            / "qrels_test.jsonl"
        )

        corpus_count = 0
        query_count = 0
        qrels_count = 0

        if corpus_path.exists():

            with open(
                corpus_path,
                "r",
                encoding="utf-8"
            ) as file:

                corpus_count = sum(
                    1
                    for _ in file
                )

        if queries_path.exists():

            with open(
                queries_path,
                "r",
                encoding="utf-8"
            ) as file:

                query_count = sum(
                    1
                    for _ in file
                )

        if qrels_path.exists():

            with open(
                qrels_path,
                "r",
                encoding="utf-8"
            ) as file:

                qrels_count = sum(
                    1
                    for _ in file
                )

        total_documents += (
            corpus_count
        )

        total_queries += (
            query_count
        )

        total_qrels += (
            qrels_count
        )

        datasets[name] = {

            "domain":
                config["domain"],

            "corpus_documents":
                corpus_count,

            "queries":
                query_count,

            "qrels":
                qrels_count,

            "corpus":
                str(
                    corpus_path
                    .relative_to(BASE_DIR)
                ),

            "queries_file":
                str(
                    queries_path
                    .relative_to(BASE_DIR)
                ),

            "qrels_file":
                str(
                    qrels_path
                    .relative_to(BASE_DIR)
                ),
        }

    manifest = {

        "dataset_version":
            "v2",

        "description":
            (
                "Multi-domain retrieval "
                "benchmark using public "
                "BEIR datasets."
            ),

        "datasets":
            datasets,

        "totals":
            {
                "documents":
                    total_documents,

                "queries":
                    total_queries,

                "qrels":
                    total_qrels,
            },
    }

    manifest_path = (
        PROCESSED_DIR
        / "dataset_manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2
        )

    print()
    print("=" * 70)
    print("MANIFEST CREATED")
    print("=" * 70)

    print(
        f"Documents : {total_documents}"
    )

    print(
        f"Queries   : {total_queries}"
    )

    print(
        f"Qrels     : {total_qrels}"
    )

    print(
        f"Manifest  : {manifest_path}"
    )


def main():

    print()
    print("=" * 70)
    print("MULTI-DOMAIN BEIR DATASET BUILDER")
    print("HUGGING FACE VERSION")
    print("=" * 70)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for name, config in DATASETS.items():

        try:

            process_dataset(
                name,
                config
            )

        except Exception as error:

            print()
            print(
                f"[ERROR] Dataset failed: "
                f"{name}"
            )

            print(
                f"Reason: {error}"
            )

            raise

    create_manifest()

    print()
    print("=" * 70)
    print("DATASET BUILD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
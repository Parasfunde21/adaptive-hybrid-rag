from pathlib import Path
import pandas as pd

from training.generate_queries import generator
from training.label_generator import label_generator

from services.feature_extractor import feature_extractor
from services.bm25_service import bm25_search
from services.retrieval_service import dense_search


def generate_training_dataset():

    dataset = []

    queries = generator.generate_queries()

    print(f"Generating labels for {len(queries)} queries...\n")

    for sample in queries:

        query = sample["query"]
        source_document = sample["source_document"]

        features = feature_extractor.extract(query)

        # BM25
        bm25_docs, bm25_scores, _ = bm25_search(
            query,
            top_k=10
        )

        # Dense
        dense_docs, dense_distances, _ = dense_search(
            query,
            top_k=10
        )

        bm25_norm, dense_norm = label_generator.quality_scores(
            bm25_scores,
            dense_distances
        )

        bm25_rank = None
        dense_rank = None

        bm25_score = 0
        dense_score = 0

        if source_document in bm25_docs:

            bm25_rank = bm25_docs.index(source_document) + 1
            bm25_score = bm25_norm[bm25_rank - 1]

        if source_document in dense_docs:

            dense_rank = dense_docs.index(source_document) + 1
            dense_score = dense_norm[dense_rank - 1]

        bm25_weight, dense_weight = label_generator.adaptive_weights(
            bm25_rank,
            dense_rank,
            bm25_score,
            dense_score
        )

        row = features.copy()

        row["bm25_weight"] = bm25_weight
        row["dense_weight"] = dense_weight

        dataset.append(row)

    df = pd.DataFrame(dataset)

    dataset_path = Path(__file__).parent / "training_dataset.csv"

    df.to_csv(
        dataset_path,
        index=False
    )

    print(df.head())

    print()

    print(f"Generated {len(df)} training samples")

    print()

    print(f"Saved to: {dataset_path}")


if __name__ == "__main__":
    generate_training_dataset()
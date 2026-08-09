import time
import pandas as pd
import numpy as np

from services.bm25_service import bm25_search
from services.retrieval_service import dense_search
from services.rrf_service import rrf
from services.fusion_service import fusion_service
from services.adaptive_predictor_v3 import adaptive_predictor_v3


TEST_PATH = "training/ml_test.csv"

TOP_K = 10


# =========================================================
# Utility
# =========================================================

def normalize(scores):

    scores = np.asarray(
        scores,
        dtype=float
    )

    if len(scores) == 0:
        return scores

    minimum = scores.min()
    maximum = scores.max()

    if maximum == minimum:
        return np.ones_like(scores)

    return (
        scores - minimum
    ) / (
        maximum - minimum
    )


def dense_similarity(distances):

    return np.array([
        1.0 / (1.0 + float(distance))
        for distance in distances
    ])


# =========================================================
# Retrieve BM25
# =========================================================

def get_bm25(query):

    documents, scores, _ = bm25_search(
        query=query,
        top_k=TOP_K
    )

    return documents, scores


# =========================================================
# Retrieve Dense
# =========================================================

def get_dense(query):

    documents, distances, _ = dense_search(
        query=query,
        top_k=TOP_K
    )

    return documents, distances


# =========================================================
# Adaptive V3
# =========================================================

def get_adaptive_v3(query):

    # =================================================
    # 1. Retrieve once
    # =================================================

    dense_docs, dense_distances = (
        get_dense(query)
    )

    bm25_docs, bm25_scores = (
        get_bm25(query)
    )

    # =================================================
    # 2. Extract retrieval features
    # =================================================

    from services.retrieval_feature_service import (
        retrieval_feature_service
    )

    retrieval_features = (
        retrieval_feature_service.extract(

            bm25_scores=bm25_scores,

            dense_distances=dense_distances,

            bm25_docs=bm25_docs,

            dense_docs=dense_docs

        )
    )

    # =================================================
    # 3. Predict V3 weights
    # =================================================

    prediction = (
        adaptive_predictor_v3.predict(

            query=query,

            retrieval_features=
                retrieval_features

        )
    )

    bm25_weight = prediction[
        "bm25_weight"
    ]

    dense_weight = prediction[
        "dense_weight"
    ]

    # =================================================
    # 4. Fuse
    # =================================================

    results = fusion_service.fuse(

        dense_docs=dense_docs,

        dense_distances=dense_distances,

        bm25_docs=bm25_docs,

        bm25_scores=bm25_scores,

        dense_weight=dense_weight,

        bm25_weight=bm25_weight

    )

    return results[:TOP_K]
# =========================================================
# Existing Adaptive
# =========================================================

def get_existing_adaptive(query):

    from services.hybrid_service import hybrid_search

    response = hybrid_search(
        query=query,
        top_k=TOP_K
    )

    return response["results"]


# =========================================================
# RRF
# =========================================================

def get_rrf(query):

    dense_docs, _ = get_dense(query)

    bm25_docs, _ = get_bm25(query)

    return rrf.fuse(
        dense_docs,
        bm25_docs,
        
    )[:TOP_K]


# =========================================================
# Evaluate Retrieval
# =========================================================

def evaluate_results(
    results,
    source_document
):

    retrieved_documents = [

        item["document"]

        if isinstance(item, dict)

        else item

        for item in results

    ]

    relevant = [

        source_document

    ]

    # -----------------------------------------------------
    # Precision@K
    # -----------------------------------------------------

    relevant_count = sum(

        1

        for document
        in retrieved_documents

        if document == source_document

    )

    precision = (
        relevant_count
        /
        TOP_K
    )

    # -----------------------------------------------------
    # Recall@K
    # -----------------------------------------------------

    recall = (

        1.0
        if source_document
        in retrieved_documents

        else 0.0

    )

    # -----------------------------------------------------
    # MRR
    # -----------------------------------------------------

    mrr = 0.0

    for rank, document in enumerate(
        retrieved_documents,
        start=1
    ):

        if document == source_document:

            mrr = 1.0 / rank

            break

    # -----------------------------------------------------
    # nDCG@K
    # -----------------------------------------------------

    dcg = 0.0

    for rank, document in enumerate(
        retrieved_documents,
        start=1
    ):

        if document == source_document:

            dcg = 1.0 / np.log2(
                rank + 1
            )

            break

    ideal_dcg = 1.0

    ndcg = (
        dcg / ideal_dcg
    )

    return {
        "precision": precision,
        "recall": recall,
        "mrr": mrr,
        "ndcg": ndcg
    }


# =========================================================
# Main
# =========================================================

def main():

    print()
    print("=" * 80)
    print("V3 ADAPTIVE RETRIEVAL BENCHMARK")
    print("=" * 80)

    test_df = pd.read_csv(
        TEST_PATH
    )

    print(
        f"\nTest queries : "
        f"{len(test_df)}"
    )

    methods = [

        "BM25",

        "Dense",

        "RRF",

        "Existing Adaptive",

        "V3 Adaptive"

    ]

    metrics = {

        method: {

            "precision": [],
            "recall": [],
            "mrr": [],
            "ndcg": [],
            "latency": []

        }

        for method in methods

    }

    # =====================================================
    # Evaluation loop
    # =====================================================

    for index, row in test_df.iterrows():

        query = row["query"]

        source_document = row[
            "source_document"
        ]

        if (
            index == 0
            or (index + 1) % 25 == 0
            or index + 1 == len(test_df)
        ):

            print(
                f"Processed "
                f"{index + 1}/"
                f"{len(test_df)}"
            )

        # -------------------------------------------------
        # BM25
        # -------------------------------------------------

        start = time.perf_counter()

        bm25_docs, bm25_scores = (
            get_bm25(query)
        )

        latency = (
            time.perf_counter()
            - start
        )

        result = evaluate_results(
            [
                {
                    "document": doc
                }

                for doc in bm25_docs
            ],
            source_document
        )

        for key in [
            "precision",
            "recall",
            "mrr",
            "ndcg"
        ]:

            metrics["BM25"][key].append(
                result[key]
            )

        metrics["BM25"][
            "latency"
        ].append(latency)

        # -------------------------------------------------
        # Dense
        # -------------------------------------------------

        start = time.perf_counter()

        dense_docs, dense_distances = (
            get_dense(query)
        )

        latency = (
            time.perf_counter()
            - start
        )

        result = evaluate_results(
            [
                {
                    "document": doc
                }

                for doc in dense_docs
            ],
            source_document
        )

        for key in [
            "precision",
            "recall",
            "mrr",
            "ndcg"
        ]:

            metrics["Dense"][key].append(
                result[key]
            )

        metrics["Dense"][
            "latency"
        ].append(latency)

        # -------------------------------------------------
        # RRF
        # -------------------------------------------------

        start = time.perf_counter()

        rrf_results = get_rrf(
            query
        )

        latency = (
            time.perf_counter()
            - start
        )

        result = evaluate_results(
            rrf_results,
            source_document
        )

        for key in [
            "precision",
            "recall",
            "mrr",
            "ndcg"
        ]:

            metrics["RRF"][key].append(
                result[key]
            )

        metrics["RRF"][
            "latency"
        ].append(latency)

        # -------------------------------------------------
        # Existing Adaptive
        # -------------------------------------------------

        start = time.perf_counter()

        existing_results = (
            get_existing_adaptive(
                query
            )
        )

        latency = (
            time.perf_counter()
            - start
        )

        result = evaluate_results(
            existing_results,
            source_document
        )

        for key in [
            "precision",
            "recall",
            "mrr",
            "ndcg"
        ]:

            metrics[
                "Existing Adaptive"
            ][key].append(
                result[key]
            )

        metrics[
            "Existing Adaptive"
        ]["latency"].append(
            latency
        )

        # -------------------------------------------------
        # V3 Adaptive
        # -------------------------------------------------

        start = time.perf_counter()

        v3_results = (
            get_adaptive_v3(
                query
            )
        )

        latency = (
            time.perf_counter()
            - start
        )

        result = evaluate_results(
            v3_results,
            source_document
        )

        for key in [
            "precision",
            "recall",
            "mrr",
            "ndcg"
        ]:

            metrics[
                "V3 Adaptive"
            ][key].append(
                result[key]
            )

        metrics[
            "V3 Adaptive"
        ]["latency"].append(
            latency
        )

    # =====================================================
    # Summary
    # =====================================================

    print()
    print("=" * 100)
    print("FINAL RETRIEVAL COMPARISON")
    print("=" * 100)

    print(
        f"{'Method':<22}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'MRR':>12}"
        f"{'nDCG':>12}"
        f"{'Latency':>12}"
    )

    print("-" * 82)

    summary_rows = []

    for method in methods:

        precision = np.mean(
            metrics[method]["precision"]
        )

        recall = np.mean(
            metrics[method]["recall"]
        )

        mrr = np.mean(
            metrics[method]["mrr"]
        )

        ndcg = np.mean(
            metrics[method]["ndcg"]
        )

        latency = np.mean(
            metrics[method]["latency"]
        )

        print(
            f"{method:<22}"
            f"{precision:>12.4f}"
            f"{recall:>12.4f}"
            f"{mrr:>12.4f}"
            f"{ndcg:>12.4f}"
            f"{latency:>12.4f}"
        )

        summary_rows.append({

            "method": method,

            "precision_at_10":
                precision,

            "recall_at_10":
                recall,

            "mrr":
                mrr,

            "ndcg_at_10":
                ndcg,

            "latency":
                latency

        })

    # =====================================================
    # Save
    # =====================================================

    output = pd.DataFrame(
        summary_rows
    )

    output_path = (
        "evaluation/v3_retrieval_results.csv"
    )

    output.to_csv(
        output_path,
        index=False
    )

    print()
    print(
        f"Saved results to: "
        f"{output_path}"
    )

    print()


if __name__ == "__main__":
    main()
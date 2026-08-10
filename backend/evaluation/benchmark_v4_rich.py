import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from services.bm25_service import bm25_search
from services.retrieval_service import dense_search
from services.fusion_service import fusion_service
from services.hybrid_service import hybrid_search
from services.adaptive_predictor_v3 import adaptive_predictor_v3
from services.retrieval_feature_service import retrieval_feature_service


BASE_DIR = Path(__file__).resolve().parents[1]

TEST_PATH = BASE_DIR / "training" / "ml_test.csv"
V4_PREDICTIONS_PATH = BASE_DIR / "evaluation" / "v4_analysis" / "v4_rich_test_predictions.csv"
V4_MODEL_PATH = BASE_DIR / "models" / "adaptive_weight_model_v4_rich.pkl"
OUTPUT_DIR = BASE_DIR / "evaluation" / "v4_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 10


def rrf_fuse(dense_docs, bm25_docs, top_k=10, k=60):
    scores = {}
    order = {}

    for rank, doc in enumerate(dense_docs, start=1):
        scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
        order.setdefault(doc, len(order))

    for rank, doc in enumerate(bm25_docs, start=1):
        scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
        order.setdefault(doc, len(order))

    ranked = sorted(scores.items(), key=lambda x: (-x[1], order[x[0]]))
    return [{"document": doc, "fusion_score": score} for doc, score in ranked[:top_k]]


def evaluate_results(results, source_document):
    documents = []
    for item in results:
        if isinstance(item, dict):
            documents.append(item.get("document", ""))
        else:
            documents.append(item)

    precision = sum(1 for document in documents if document == source_document) / TOP_K
    recall = 1.0 if source_document in documents else 0.0

    mrr = 0.0
    ndcg = 0.0
    for rank, document in enumerate(documents, start=1):
        if document == source_document:
            mrr = 1.0 / rank
            ndcg = 1.0 / np.log2(rank + 1)
            break

    return {
        "precision": precision,
        "recall": recall,
        "mrr": mrr,
        "ndcg": ndcg,
    }


def get_v3(query):
    dense_docs, dense_distances, dense_ids, dense_metadatas = dense_search(query=query, top_k=TOP_K)
    bm25_docs, bm25_scores, bm25_ids, bm25_metadatas = bm25_search(query=query, top_k=TOP_K)

    features = retrieval_feature_service.extract(
        query,
        bm25_scores,
        dense_distances,
        bm25_docs,
        dense_docs,
    )

    prediction = adaptive_predictor_v3.predict(
        query=query,
        retrieval_features=features,
    )

    results = fusion_service.fuse(
        dense_docs=dense_docs,
        dense_distances=dense_distances,
        dense_ids=dense_ids,
        dense_metadatas=dense_metadatas,
        bm25_docs=bm25_docs,
        bm25_scores=bm25_scores,
        bm25_ids=bm25_ids,
        bm25_metadatas=bm25_metadatas,
        dense_weight=prediction["dense_weight"],
        bm25_weight=prediction["bm25_weight"],
    )

    return results[:TOP_K], prediction["bm25_weight"]


def tokenize(text):
    import re
    return re.findall(r"\b\w+\b", str(text).lower())


def lexical_coverage(query, document):
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    document_tokens = set(tokenize(document))
    return len(query_tokens & document_tokens) / len(query_tokens)


def build_rich_features(query, bm25_docs, bm25_scores, dense_docs, dense_distances):
    bm25_scores = np.asarray(bm25_scores, dtype=float)
    dense_distances = np.asarray(dense_distances, dtype=float)
    dense_similarity = 1.0 - dense_distances

    if len(bm25_scores):
        bm25_top = float(bm25_scores[0])
        bm25_mean = float(np.mean(bm25_scores))
        bm25_std = float(np.std(bm25_scores))
        bm25_gap = bm25_top - float(bm25_scores[1]) if len(bm25_scores) > 1 else bm25_top
        bm25_ratio = bm25_top / bm25_mean if bm25_mean != 0 else 0.0
    else:
        bm25_top = bm25_mean = bm25_std = bm25_gap = bm25_ratio = 0.0

    if len(dense_similarity):
        dense_top = float(dense_similarity[0])
        dense_mean = float(np.mean(dense_similarity))
        dense_std = float(np.std(dense_similarity))
        dense_gap = dense_top - float(dense_similarity[1]) if len(dense_similarity) > 1 else dense_top
        dense_ratio = dense_top / dense_mean if dense_mean != 0 else 0.0
    else:
        dense_top = dense_mean = dense_std = dense_gap = dense_ratio = 0.0

    bm25_set = set(bm25_docs)
    dense_set = set(dense_docs)
    union = bm25_set | dense_set
    intersection = bm25_set & dense_set
    overlap = len(intersection) / len(union) if union else 0.0

    common = bm25_set & dense_set
    if len(common) >= 2:
        bm25_rank = {doc: rank for rank, doc in enumerate(bm25_docs)}
        dense_rank = {doc: rank for rank, doc in enumerate(dense_docs)}
        differences = [abs(bm25_rank[doc] - dense_rank[doc]) for doc in common]
        rank_agreement = 1.0 - (np.mean(differences) / (TOP_K - 1))
    else:
        rank_agreement = 0.0

    tokens = tokenize(query)
    unique_tokens = set(tokens)
    query_count = len(tokens)
    unique_count = len(unique_tokens)
    avg_length = sum(len(token) for token in tokens) / query_count if query_count else 0.0

    bm25_coverage = lexical_coverage(query, bm25_docs[0]) if bm25_docs else 0.0
    dense_coverage = lexical_coverage(query, dense_docs[0]) if dense_docs else 0.0

    return {
        "bm25_top_score": bm25_top,
        "bm25_mean_score": bm25_mean,
        "bm25_score_std": bm25_std,
        "bm25_score_gap": bm25_gap,
        "bm25_top_mean_ratio": bm25_ratio,
        "dense_top_similarity": dense_top,
        "dense_mean_similarity": dense_mean,
        "dense_similarity_std": dense_std,
        "dense_similarity_gap": dense_gap,
        "dense_top_mean_ratio": dense_ratio,
        "overlap_ratio": overlap,
        "rank_agreement": rank_agreement,
        "query_token_count": query_count,
        "query_unique_token_count": unique_count,
        "query_avg_token_length": avg_length,
        "bm25_top_lexical_coverage": bm25_coverage,
        "dense_top_lexical_coverage": dense_coverage,
    }


def get_v4_rich(query, model, prediction_map):
    dense_docs, dense_distances, dense_ids, dense_metadatas = dense_search(query=query, top_k=TOP_K)
    bm25_docs, bm25_scores, bm25_ids, bm25_metadatas = bm25_search(query=query, top_k=TOP_K)

    if query in prediction_map:
        bm25_weight = float(prediction_map[query])
    else:
        features = build_rich_features(query, bm25_docs, bm25_scores, dense_docs, dense_distances)
        feature_order = [
            "bm25_top_score",
            "bm25_mean_score",
            "bm25_score_std",
            "bm25_score_gap",
            "bm25_top_mean_ratio",
            "dense_top_similarity",
            "dense_mean_similarity",
            "dense_similarity_std",
            "dense_similarity_gap",
            "dense_top_mean_ratio",
            "overlap_ratio",
            "rank_agreement",
            "query_token_count",
            "query_unique_token_count",
            "query_avg_token_length",
            "bm25_top_lexical_coverage",
            "dense_top_lexical_coverage",
        ]
        X = pd.DataFrame([[features[name] for name in feature_order]], columns=feature_order)
        bm25_weight = float(model.predict(X)[0])

    bm25_weight = float(np.clip(bm25_weight, 0.0, 1.0))
    dense_weight = 1.0 - bm25_weight

    results = fusion_service.fuse(
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

    return results[:TOP_K], bm25_weight


def main():
    print()
    print("=" * 100)
    print("V4 RICH HELD-OUT RETRIEVAL BENCHMARK")
    print("=" * 100)

    test_df = pd.read_csv(TEST_PATH)
    predictions_df = pd.read_csv(V4_PREDICTIONS_PATH)

    print()
    print(f"Full test dataset rows: {len(test_df)}")
    print(f"V4 held-out predictions: {len(predictions_df)}")

    if "query" not in predictions_df.columns:
        raise ValueError("V4 prediction file does not contain query column.")

    test_queries = set(predictions_df["query"].astype(str))
    evaluation_df = test_df[test_df["query"].astype(str).isin(test_queries)].copy()

    print()
    print(f"Evaluation queries: {len(evaluation_df)}")

    prediction_map = dict(
        zip(
            predictions_df["query"].astype(str),
            predictions_df["predicted_bm25_weight"].astype(float),
        )
    )

    model = joblib.load(V4_MODEL_PATH)

    print()
    print(f"V4 model loaded:\n{V4_MODEL_PATH}")

    methods = [
        "BM25",
        "Dense",
        "RRF",
        "Existing Adaptive",
        "V3 Adaptive",
        "V4 Rich Adaptive",
    ]

    metrics = {
        method: {"precision": [], "recall": [], "mrr": [], "ndcg": [], "latency": []}
        for method in methods
    }

    v4_weights = []

    for processed, (_, row) in enumerate(
    evaluation_df.iterrows(),
    start=1
    ):
        query = str(row["query"])
        source_document = row["source_document"]

        print(f"Processed {processed}/{len(evaluation_df)}")

        start = time.perf_counter()
        bm25_docs, bm25_scores, bm25_ids, bm25_metadatas = bm25_search(query=query, top_k=TOP_K)
        latency = time.perf_counter() - start
        result = evaluate_results([{"document": doc} for doc in bm25_docs], source_document)
        for key in result:
            metrics["BM25"][key].append(result[key])
        metrics["BM25"]["latency"].append(latency)

        start = time.perf_counter()
        dense_docs, dense_distances, dense_ids, dense_metadatas = dense_search(query=query, top_k=TOP_K)
        latency = time.perf_counter() - start
        result = evaluate_results([{"document": doc} for doc in dense_docs], source_document)
        for key in result:
            metrics["Dense"][key].append(result[key])
        metrics["Dense"]["latency"].append(latency)

        start = time.perf_counter()
        rrf_results = rrf_fuse(dense_docs=dense_docs, bm25_docs=bm25_docs, top_k=TOP_K)
        latency = time.perf_counter() - start
        result = evaluate_results(rrf_results, source_document)
        for key in result:
            metrics["RRF"][key].append(result[key])
        metrics["RRF"]["latency"].append(latency)

        start = time.perf_counter()
        existing_response = hybrid_search(query=query, top_k=TOP_K)
        existing_results = existing_response["results"]
        latency = time.perf_counter() - start
        result = evaluate_results(existing_results, source_document)
        for key in result:
            metrics["Existing Adaptive"][key].append(result[key])
        metrics["Existing Adaptive"]["latency"].append(latency)

        start = time.perf_counter()
        v3_results, v3_weight = get_v3(query)
        latency = time.perf_counter() - start
        result = evaluate_results(v3_results, source_document)
        for key in result:
            metrics["V3 Adaptive"][key].append(result[key])
        metrics["V3 Adaptive"]["latency"].append(latency)

        start = time.perf_counter()
        v4_results, v4_weight = get_v4_rich(query=query, model=model, prediction_map=prediction_map)
        latency = time.perf_counter() - start
        result = evaluate_results(v4_results, source_document)
        for key in result:
            metrics["V4 Rich Adaptive"][key].append(result[key])
        metrics["V4 Rich Adaptive"]["latency"].append(latency)

        v4_weights.append(
            {
                "query": query,
                "v3_bm25_weight": v3_weight,
                "v4_bm25_weight": v4_weight,
                "source_document": source_document,
                "v3_mrr": metrics["V3 Adaptive"]["mrr"][-1],
                "v4_mrr": metrics["V4 Rich Adaptive"]["mrr"][-1],
            }
        )

    print()
    print("=" * 100)
    print("FINAL HELD-OUT RETRIEVAL COMPARISON")
    print("=" * 100)

    print(f"{'Method':<24}{'Precision':>12}{'Recall':>12}{'MRR':>12}{'nDCG':>12}{'Latency':>14}")
    print("-" * 86)

    summary_rows = []
    for method in methods:
        precision = np.mean(metrics[method]["precision"])
        recall = np.mean(metrics[method]["recall"])
        mrr = np.mean(metrics[method]["mrr"])
        ndcg = np.mean(metrics[method]["ndcg"])
        latency = np.mean(metrics[method]["latency"])

        print(f"{method:<24}{precision:>12.4f}{recall:>12.4f}{mrr:>12.4f}{ndcg:>12.4f}{latency:>14.4f}")

        summary_rows.append(
            {
                "method": method,
                "precision_at_10": precision,
                "recall_at_10": recall,
                "mrr": mrr,
                "ndcg_at_10": ndcg,
                "latency": latency,
            }
        )

    summary = pd.DataFrame(summary_rows)

    v3_mrr = float(summary.loc[summary["method"] == "V3 Adaptive", "mrr"].iloc[0])
    v4_mrr = float(summary.loc[summary["method"] == "V4 Rich Adaptive", "mrr"].iloc[0])
    v3_ndcg = float(summary.loc[summary["method"] == "V3 Adaptive", "ndcg_at_10"].iloc[0])
    v4_ndcg = float(summary.loc[summary["method"] == "V4 Rich Adaptive", "ndcg_at_10"].iloc[0])

    print()
    print("=" * 100)
    print("V3 → V4 IMPROVEMENT")
    print("=" * 100)
    print(f"MRR improvement  : {v4_mrr - v3_mrr:+.4f}")
    print(f"nDCG improvement : {v4_ndcg - v3_ndcg:+.4f}")
    if v3_mrr != 0:
        print(f"MRR relative %   : {((v4_mrr - v3_mrr) / v3_mrr) * 100:+.2f}%")

    summary_path = OUTPUT_DIR / "v4_heldout_retrieval_results.csv"
    summary.to_csv(summary_path, index=False)

    weights_df = pd.DataFrame(v4_weights)
    weights_path = OUTPUT_DIR / "v4_vs_v3_query_comparison.csv"
    weights_df.to_csv(weights_path, index=False)

    print()
    print(f"Saved summary:\n{summary_path}")
    print()
    print(f"Saved query comparison:\n{weights_path}")
    print()
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
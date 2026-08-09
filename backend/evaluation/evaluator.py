import time

from evaluation.dataset import dataset
from evaluation.metrics import metrics

from services.bm25_service import bm25_search
from services.retrieval_service import dense_search
from services.hybrid_service import hybrid_search
from services.rrf_service import rrf


class RetrievalEvaluator:

    # -------------------------------------------------
    # BM25 Evaluation
    # -------------------------------------------------

    def evaluate_bm25(
        self,
        top_k=10
    ):

        samples = dataset.load()

        precision = []
        recall = []
        mrr = []
        ndcg = []
        latency = []

        print(f"\nEvaluating BM25 on {len(samples)} queries...\n")

        for sample in samples:

            query = sample["query"]

            relevant = {
                sample["source_document"]
            }

            start = time.perf_counter()

            docs, _, _ = bm25_search(
                query,
                top_k=top_k
            )

            latency.append(
                time.perf_counter() - start
            )

            precision.append(
                metrics.precision_at_k(
                    docs,
                    relevant,
                    top_k
                )
            )

            recall.append(
                metrics.recall_at_k(
                    docs,
                    relevant,
                    top_k
                )
            )

            mrr.append(
                metrics.reciprocal_rank(
                    docs,
                    relevant
                )
            )

            ndcg.append(
                metrics.ndcg_at_k(
                    docs,
                    relevant,
                    top_k
                )
            )

        return {

            "precision": sum(precision) / len(precision),

            "recall": sum(recall) / len(recall),

            "mrr": sum(mrr) / len(mrr),

            "ndcg": sum(ndcg) / len(ndcg),

            "latency": sum(latency) / len(latency)

        }

    # -------------------------------------------------
    # Dense Retrieval Evaluation
    # -------------------------------------------------

    def evaluate_dense(
        self,
        top_k=10
    ):

        samples = dataset.load()

        precision = []
        recall = []
        mrr = []
        ndcg = []
        latency = []

        print("\nEvaluating Dense Retrieval...\n")

        for sample in samples:

            query = sample["query"]

            relevant = {
                sample["source_document"]
            }

            start = time.perf_counter()

            docs, _, _ = dense_search(
                query,
                top_k=top_k
            )

            latency.append(
                time.perf_counter() - start
            )

            precision.append(
                metrics.precision_at_k(
                    docs,
                    relevant,
                    top_k
                )
            )

            recall.append(
                metrics.recall_at_k(
                    docs,
                    relevant,
                    top_k
                )
            )

            mrr.append(
                metrics.reciprocal_rank(
                    docs,
                    relevant
                )
            )

            ndcg.append(
                metrics.ndcg_at_k(
                    docs,
                    relevant,
                    top_k
                )
            )

        return {

            "precision": sum(precision) / len(precision),

            "recall": sum(recall) / len(recall),

            "mrr": sum(mrr) / len(mrr),

            "ndcg": sum(ndcg) / len(ndcg),

            "latency": sum(latency) / len(latency)

        }

    # -------------------------------------------------
    # Reciprocal Rank Fusion Evaluation
    # -------------------------------------------------

    def evaluate_rrf(
        self,
        top_k=10
    ):

        samples = dataset.load()

        precision = []
        recall = []
        mrr = []
        ndcg = []
        latency = []

        print("\nEvaluating Reciprocal Rank Fusion...\n")

        for sample in samples:

            query = sample["query"]

            relevant = {
                sample["source_document"]
            }

            start = time.perf_counter()

            dense_docs, _, _ = dense_search(
                query,
                top_k=top_k
            )

            bm25_docs, _, _ = bm25_search(
                query,
                top_k=top_k
            )

            results = rrf.fuse(
                dense_docs,
                bm25_docs
            )

            docs = [

                item["document"]

                for item in results[:top_k]

            ]

            latency.append(
                time.perf_counter() - start
            )

            precision.append(
                metrics.precision_at_k(
                    docs,
                    relevant,
                    top_k
                )
            )

            recall.append(
                metrics.recall_at_k(
                    docs,
                    relevant,
                    top_k
                )
            )

            mrr.append(
                metrics.reciprocal_rank(
                    docs,
                    relevant
                )
            )

            ndcg.append(
                metrics.ndcg_at_k(
                    docs,
                    relevant,
                    top_k
                )
            )

        return {

            "precision": sum(precision) / len(precision),

            "recall": sum(recall) / len(recall),

            "mrr": sum(mrr) / len(mrr),

            "ndcg": sum(ndcg) / len(ndcg),

            "latency": sum(latency) / len(latency)

        }

    # -------------------------------------------------
    # Adaptive Hybrid Evaluation
    # -------------------------------------------------

    def evaluate_adaptive(
        self,
        top_k=10
    ):

        samples = dataset.load()

        precision = []
        recall = []
        mrr = []
        ndcg = []
        latency = []

        print("\nEvaluating Adaptive Hybrid...\n")

        for sample in samples:

            query = sample["query"]

            relevant = {
                sample["source_document"]
            }

            start = time.perf_counter()

            results = hybrid_search(
                query,
                top_k=top_k
            )["results"]

            docs = [

                item["document"]

                for item in results

            ]

            latency.append(
                time.perf_counter() - start
            )

            precision.append(
                metrics.precision_at_k(
                    docs,
                    relevant,
                    top_k
                )
            )

            recall.append(
                metrics.recall_at_k(
                    docs,
                    relevant,
                    top_k
                )
            )

            mrr.append(
                metrics.reciprocal_rank(
                    docs,
                    relevant
                )
            )

            ndcg.append(
                metrics.ndcg_at_k(
                    docs,
                    relevant,
                    top_k
                )
            )

        return {

            "precision": sum(precision) / len(precision),

            "recall": sum(recall) / len(recall),

            "mrr": sum(mrr) / len(mrr),

            "ndcg": sum(ndcg) / len(ndcg),

            "latency": sum(latency) / len(latency)

        }


evaluator = RetrievalEvaluator()
import time
import pandas as pd

from evaluation.dataset import dataset
from evaluation.metrics import metrics

from services.bm25_service import bm25_search
from services.retrieval_service import dense_search
from services.fusion_service import AdaptiveScoreFusion


class PerformanceBasedWeightOptimizer:

    """
    Generates training targets using actual retrieval performance.

    For every query:

        1. Run BM25 retrieval.
        2. Run Dense retrieval.
        3. Try multiple BM25/Dense weight combinations.
        4. Evaluate every combination.
        5. Select the weight producing the best retrieval quality.

    The selected BM25 weight becomes the ML training target.

    This replaces heuristic label generation.
    """

    def __init__(
        self,
        top_k=10,
        weight_step=0.05
    ):

        self.top_k = top_k

        self.weight_step = weight_step

        self.fusion_service = AdaptiveScoreFusion()

    # -------------------------------------------------
    # Generate Weight Combinations
    # -------------------------------------------------

    def generate_weights(self):

        weights = []

        current = 0.0

        while current <= 1.000001:

            bm25_weight = round(
                current,
                2
            )

            dense_weight = round(
                1.0 - bm25_weight,
                2
            )

            weights.append(
                (
                    bm25_weight,
                    dense_weight
                )
            )

            current += self.weight_step

        return weights

    # -------------------------------------------------
    # Evaluate One Weight Combination
    # -------------------------------------------------

    def evaluate_weight(
        self,
        dense_docs,
        dense_distances,
        bm25_docs,
        bm25_scores,
        relevant_documents,
        bm25_weight,
        dense_weight
    ):

        fused_results = self.fusion_service.fuse(

            dense_docs=dense_docs,

            dense_distances=dense_distances,

            bm25_docs=bm25_docs,

            bm25_scores=bm25_scores,

            dense_weight=dense_weight,

            bm25_weight=bm25_weight

        )

        retrieved_documents = [

            item["document"]

            for item in fused_results[:self.top_k]

        ]

        precision = metrics.precision_at_k(

            retrieved_documents,

            relevant_documents,

            self.top_k

        )

        recall = metrics.recall_at_k(

            retrieved_documents,

            relevant_documents,

            self.top_k

        )

        mrr = metrics.reciprocal_rank(

            retrieved_documents,

            relevant_documents

        )

        ndcg = metrics.ndcg_at_k(

            retrieved_documents,

            relevant_documents,

            self.top_k

        )

        return {

            "precision": precision,

            "recall": recall,

            "mrr": mrr,

            "ndcg": ndcg

        }

    # -------------------------------------------------
    # Find Best Weight For One Query
    # -------------------------------------------------

    def optimize_query(
        self,
        sample,
        weight_combinations
    ):

        query = sample["query"]

        relevant_documents = {

            sample["source_document"]

        }

        # ---------------------------------------------
        # Retrieve only once
        # ---------------------------------------------

        dense_docs, dense_distances, _ = dense_search(

            query,

            top_k=self.top_k

        )

        bm25_docs, bm25_scores, _ = bm25_search(

            query,

            top_k=self.top_k

        )

        best_result = None

        all_results = []

        # ---------------------------------------------
        # Test every weight combination
        # ---------------------------------------------

        for bm25_weight, dense_weight in weight_combinations:

            scores = self.evaluate_weight(

                dense_docs=dense_docs,

                dense_distances=dense_distances,

                bm25_docs=bm25_docs,

                bm25_scores=bm25_scores,

                relevant_documents=relevant_documents,

                bm25_weight=bm25_weight,

                dense_weight=dense_weight

            )

            result = {

                "bm25_weight": bm25_weight,

                "dense_weight": dense_weight,

                **scores

            }

            all_results.append(result)

            # -----------------------------------------
            # Optimization objective
            #
            # Primary   : nDCG
            # Secondary : MRR
            # Tertiary  : Recall
            # -----------------------------------------

            if best_result is None:

                best_result = result

            else:

                current_key = (

                    result["ndcg"],

                    result["mrr"],

                    result["recall"]

                )

                best_key = (

                    best_result["ndcg"],

                    best_result["mrr"],

                    best_result["recall"]

                )

                if current_key > best_key:

                    best_result = result

        return best_result, all_results

    # -------------------------------------------------
    # Optimize Complete Dataset
    # -------------------------------------------------

    def generate_labels(
        self,
        output_path="training/optimized_training_dataset.csv"
    ):

        samples = dataset.load()

        weight_combinations = self.generate_weights()

        print()
        print("=" * 70)
        print("PERFORMANCE-BASED WEIGHT OPTIMIZATION")
        print("=" * 70)

        print(
            f"Queries              : {len(samples)}"
        )

        print(
            f"Weight combinations  : {len(weight_combinations)}"
        )

        print(
            f"Weight step          : {self.weight_step}"
        )

        print(
            f"Optimization metric  : nDCG@{self.top_k}"
        )

        print("=" * 70)
        print()

        optimized_rows = []

        start_time = time.perf_counter()

        for index, sample in enumerate(
            samples,
            start=1
        ):

            best_result, _ = self.optimize_query(

                sample,

                weight_combinations

            )

            row = {

                "query": sample["query"],

                "source_document": sample[
                    "source_document"
                ],

                "bm25_weight": best_result[
                    "bm25_weight"
                ],

                "dense_weight": best_result[
                    "dense_weight"
                ],

                "target_precision": best_result[
                    "precision"
                ],

                "target_recall": best_result[
                    "recall"
                ],

                "target_mrr": best_result[
                    "mrr"
                ],

                "target_ndcg": best_result[
                    "ndcg"
                ]

            }

            optimized_rows.append(row)

            if (

                index == 1

                or index % 50 == 0

                or index == len(samples)

            ):

                elapsed = (

                    time.perf_counter()

                    - start_time

                )

                print(

                    f"Processed "

                    f"{index}/{len(samples)} "

                    f"queries "

                    f"({elapsed:.1f}s)"

                )

        dataframe = pd.DataFrame(
            optimized_rows
        )

        dataframe.to_csv(
            output_path,
            index=False
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        print()
        print("=" * 70)
        print("OPTIMIZATION COMPLETE")
        print("=" * 70)

        print(
            f"Generated labels : {len(dataframe)}"
        )

        print(
            f"Time taken       : {elapsed:.2f} sec"
        )

        print()
        print(
            "BM25 weight distribution:"
        )

        print(
            dataframe[
                "bm25_weight"
            ].describe()
        )

        print()
        print(
            "Saved to:"
        )

        print(output_path)


optimizer = PerformanceBasedWeightOptimizer()


if __name__ == "__main__":

    optimizer.generate_labels() 
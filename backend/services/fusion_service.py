import numpy as np


class AdaptiveScoreFusion:

    """
    Adaptive Score Fusion (ASF)

    Combines normalized BM25 scores and normalized
    Dense similarities using ML-predicted weights.
    """

    def normalize(self, scores):

        scores = np.array(scores, dtype=float)

        if len(scores) == 0:
            return scores

        maximum = scores.max()
        minimum = scores.min()

        if maximum == minimum:
            return np.ones_like(scores)

        return (scores - minimum) / (maximum - minimum)

    def dense_similarity(self, distances):

        return np.array([
            1 / (1 + d)
            for d in distances
        ])

    def fuse(

        self,

        dense_docs,
        dense_distances,

        bm25_docs,
        bm25_scores,

        dense_weight,
        bm25_weight

    ):

        dense_scores = self.normalize(

            self.dense_similarity(
                dense_distances
            )

        )

        bm25_scores = self.normalize(
            bm25_scores
        )

        results = {}

        # -----------------------------------
        # Dense Retrieval Contribution
        # -----------------------------------

        for rank, (doc, similarity) in enumerate(

            zip(dense_docs, dense_scores),

            start=1

        ):

            if doc not in results:

                results[doc] = {

                    "document": doc,

                    "fusion_score": 0.0,

                    "retrieved_by": [],

                    "dense_similarity": None,

                    "bm25_score": None,

                    "dense_rank": None,

                    "bm25_rank": None

                }

            results[doc]["fusion_score"] += (

                dense_weight *

                float(similarity)

            )

            results[doc]["dense_similarity"] = float(similarity)

            results[doc]["dense_rank"] = rank

            results[doc]["retrieved_by"].append(
                "Dense"
            )

        # -----------------------------------
        # BM25 Contribution
        # -----------------------------------

        for rank, (doc, score) in enumerate(

            zip(bm25_docs, bm25_scores),

            start=1

        ):

            if doc not in results:

                results[doc] = {

                    "document": doc,

                    "fusion_score": 0.0,

                    "retrieved_by": [],

                    "dense_similarity": None,

                    "bm25_score": None,

                    "dense_rank": None,

                    "bm25_rank": None

                }

            results[doc]["fusion_score"] += (

                bm25_weight *

                float(score)

            )

            results[doc]["bm25_score"] = float(score)

            results[doc]["bm25_rank"] = rank

            results[doc]["retrieved_by"].append(
                "BM25"
            )

        ranked = sorted(

            results.values(),

            key=lambda x: x["fusion_score"],

            reverse=True

        )

        for item in ranked:

            item["fusion_score"] = round(

                item["fusion_score"],

                6

            )

        return ranked


fusion_service = AdaptiveScoreFusion()
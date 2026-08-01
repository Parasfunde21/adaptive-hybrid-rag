import numpy as np


class AdaptiveScoreFusion:

    """
    Adaptive Score Fusion (ASF)

    Uses:
        • Normalized BM25 scores
        • Normalized Dense similarities
        • ML-predicted adaptive weights

    instead of Reciprocal Rank Fusion.
    """

    def normalize(self, scores):

        scores = np.array(scores, dtype=float)

        if len(scores) == 0:
            return scores

        maximum = scores.max()
        minimum = scores.min()

        if maximum == minimum:
            return np.ones_like(scores)

        return (
            scores - minimum
        ) / (
            maximum - minimum
        )

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

        # ----------------------------
        # Dense Contribution
        # ----------------------------

        for doc, score in zip(

            dense_docs,

            dense_scores

        ):

            if doc not in results:

                results[doc] = {

                    "score": 0,

                    "source": []

                }

            results[doc]["score"] += (

                dense_weight *

                float(score)

            )

            results[doc]["source"].append(

                "Dense"

            )

        # ----------------------------
        # BM25 Contribution
        # ----------------------------

        for doc, score in zip(

            bm25_docs,

            bm25_scores

        ):

            if doc not in results:

                results[doc] = {

                    "score": 0,

                    "source": []

                }

            results[doc]["score"] += (

                bm25_weight *

                float(score)

            )

            results[doc]["source"].append(

                "BM25"

            )

        ranked = sorted(

            results.items(),

            key=lambda x: x[1]["score"],

            reverse=True

        )

        output = []

        for doc, info in ranked:

            output.append({

                "document": doc,

                "score": round(

                    info["score"],

                    6

                ),

                "source": ", ".join(

                    info["source"]

                )

            })

        return output


fusion_service = AdaptiveScoreFusion()
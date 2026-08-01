import numpy as np


class LabelGenerator:

    def __init__(self, top_k=10):
        self.top_k = top_k

    def normalize(self, values):
        """
        Min-Max normalization.
        """

        values = np.array(values, dtype=float)

        if len(values) == 0:
            return values

        maximum = values.max()
        minimum = values.min()

        if maximum == minimum:
            return np.ones_like(values)

        return (values - minimum) / (maximum - minimum)

    def distance_to_similarity(self, distances):
        """
        Convert ChromaDB distances into similarity scores.
        """

        similarities = np.array([
            1 / (1 + d)
            for d in distances
        ])

        return similarities

    def rank_score(self, rank):
        """
        Convert retrieval rank into a score.

        Rank 1  -> 1.0
        Rank 10 -> 0.1
        """

        if rank is None:
            return 0.0

        return (self.top_k - rank + 1) / self.top_k

    def quality_scores(
        self,
        bm25_scores,
        dense_distances
    ):
        """
        Compute normalized retrieval qualities.
        """

        bm25_norm = self.normalize(
            bm25_scores
        )

        dense_similarity = self.distance_to_similarity(
            dense_distances
        )

        dense_norm = self.normalize(
            dense_similarity
        )

        return bm25_norm, dense_norm

    def adaptive_weights(
        self,
        bm25_rank,
        dense_rank,
        bm25_score,
        dense_score
    ):
        """
        Generate adaptive retrieval weights
        using Softmax over retrieval quality.

        Retrieval Quality =
            Rank Score × Normalized Retrieval Score
        """

        bm25_quality = (
            self.rank_score(bm25_rank)
            *
            bm25_score
        )

        dense_quality = (
            self.rank_score(dense_rank)
            *
            dense_score
        )

        qualities = np.array([
            bm25_quality,
            dense_quality
        ])

        # Numerical stability
        qualities = qualities - np.max(qualities)

        exp_values = np.exp(qualities)

        softmax = exp_values / np.sum(exp_values)

        bm25_weight = float(softmax[0])
        dense_weight = float(softmax[1])

        return bm25_weight, dense_weight


label_generator = LabelGenerator()


if __name__ == "__main__":

    bm25_scores = [7.5, 6.3, 5.8, 4.9, 4.0]

    dense_distances = [0.32, 0.44, 0.51, 0.63, 0.74]

    bm25_norm, dense_norm = label_generator.quality_scores(
        bm25_scores,
        dense_distances
    )

    bm25_weight, dense_weight = label_generator.adaptive_weights(
        bm25_rank=1,
        dense_rank=2,
        bm25_score=bm25_norm[0],
        dense_score=dense_norm[1]
    )

    print("\nNormalized BM25 Scores")
    print(bm25_norm)

    print("\nNormalized Dense Scores")
    print(dense_norm)

    print("\nAdaptive Weights")
    print("BM25 :", round(bm25_weight, 4))
    print("Dense:", round(dense_weight, 4))
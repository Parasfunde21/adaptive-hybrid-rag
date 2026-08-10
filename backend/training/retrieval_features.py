import numpy as np

from services.bm25_service import bm25_search
from services.retrieval_service import dense_search


class RetrievalFeatureExtractor:

    """
    Extracts features describing how BM25 and Dense
    retrieval behave for a query.

    These features do NOT use the ground-truth document.

    They are generated only from:

        Query
        BM25 retrieval results
        Dense retrieval results

    This allows the ML model to learn from the actual
    retrieval behaviour of the two retrievers.
    """

    def __init__(self, top_k=10):

        self.top_k = top_k

    # =====================================================
    # Safe Statistics
    # =====================================================

    @staticmethod
    def safe_mean(values):

        if len(values) == 0:
            return 0.0

        return float(
            np.mean(values)
        )

    @staticmethod
    def safe_std(values):

        if len(values) == 0:
            return 0.0

        return float(
            np.std(values)
        )

    @staticmethod
    def safe_max(values):

        if len(values) == 0:
            return 0.0

        return float(
            np.max(values)
        )

    @staticmethod
    def safe_min(values):

        if len(values) == 0:
            return 0.0

        return float(
            np.min(values)
        )

    # =====================================================
    # Score Gap
    # =====================================================

    @staticmethod
    def score_gap(scores):

        if len(scores) < 2:

            return 0.0

        return float(
            scores[0] - scores[1]
        )

    # =====================================================
    # Retrieval Overlap
    # =====================================================

    @staticmethod
    def overlap_ratio(
        dense_docs,
        bm25_docs
    ):

        if len(dense_docs) == 0:

            return 0.0

        if len(bm25_docs) == 0:

            return 0.0

        dense_set = set(
            dense_docs
        )

        bm25_set = set(
            bm25_docs
        )

        intersection = (
            dense_set &
            bm25_set
        )

        union = (
            dense_set |
            bm25_set
        )

        if len(union) == 0:

            return 0.0

        return float(
            len(intersection)
            /
            len(union)
        )

    # =====================================================
    # Rank Agreement
    # =====================================================

    @staticmethod
    def rank_agreement(
        dense_docs,
        bm25_docs
    ):

        dense_ranks = {

            doc: rank

            for rank, doc in enumerate(
                dense_docs,
                start=1
            )

        }

        bm25_ranks = {

            doc: rank

            for rank, doc in enumerate(
                bm25_docs,
                start=1
            )

        }

        common_docs = (

            set(dense_ranks.keys())

            &

            set(bm25_ranks.keys())

        )

        if len(common_docs) == 0:

            return 0.0

        differences = []

        for doc in common_docs:

            differences.append(

                abs(

                    dense_ranks[doc]

                    -

                    bm25_ranks[doc]

                )

            )

        maximum_difference = max(
            len(dense_docs),
            len(bm25_docs)
        )

        if maximum_difference == 0:

            return 0.0

        average_difference = (
            np.mean(differences)
        )

        return float(

            1.0
            -

            (
                average_difference
                /
                maximum_difference
            )

        )

    # =====================================================
    # Extract
    # =====================================================

    def extract(self, query):

        # -------------------------------------------------
        # Run BM25
        # -------------------------------------------------

        bm25_docs, bm25_scores, _ = (

            bm25_search(

                query=query,

                top_k=self.top_k

            )

        )

        # -------------------------------------------------
        # Run Dense
        # -------------------------------------------------

        dense_docs, dense_distances, _ = (

            dense_search(

                query=query,

                top_k=self.top_k

            )

        )

        # -------------------------------------------------
        # Convert Dense distances to similarities
        # -------------------------------------------------

        dense_similarities = [

            1.0 / (
                1.0 + float(distance)
            )

            for distance
            in dense_distances

        ]

        bm25_scores = [

            float(score)

            for score
            in bm25_scores

        ]

        dense_similarities = [

            float(score)

            for score
            in dense_similarities

        ]

        # -------------------------------------------------
        # Basic retrieval features
        # -------------------------------------------------

        features = {

            # -----------------------------
            # BM25
            # -----------------------------

            "bm25_top_score":

                self.safe_max(
                    bm25_scores
                ),

            "bm25_mean_score":

                self.safe_mean(
                    bm25_scores
                ),

            "bm25_std_score":

                self.safe_std(
                    bm25_scores
                ),

            "bm25_min_score":

                self.safe_min(
                    bm25_scores
                ),

            "bm25_score_gap":

                self.score_gap(
                    bm25_scores
                ),

            # -----------------------------
            # Dense
            # -----------------------------

            "dense_top_similarity":

                self.safe_max(
                    dense_similarities
                ),

            "dense_mean_similarity":

                self.safe_mean(
                    dense_similarities
                ),

            "dense_std_similarity":

                self.safe_std(
                    dense_similarities
                ),

            "dense_min_similarity":

                self.safe_min(
                    dense_similarities
                ),

            "dense_similarity_gap":

                self.score_gap(
                    dense_similarities
                ),

            # -----------------------------
            # Retriever agreement
            # -----------------------------

            "retrieval_overlap_ratio":

                self.overlap_ratio(

                    dense_docs,

                    bm25_docs

                ),

            "retrieval_rank_agreement":

                self.rank_agreement(

                    dense_docs,

                    bm25_docs

                ),

            # -----------------------------
            # Retrieval result sizes
            # -----------------------------

            "bm25_result_count":

                len(bm25_docs),

            "dense_result_count":

                len(dense_docs)

        }

        return features


retrieval_feature_extractor = (
    RetrievalFeatureExtractor()
)


# =========================================================
# Manual Test
# =========================================================

if __name__ == "__main__":

    query = input(
        "\nQuery : "
    )

    features = (
        retrieval_feature_extractor.extract(
            query
        )
    )

    print()
    print("=" * 70)
    print("RETRIEVAL FEATURES")
    print("=" * 70)

    for key, value in features.items():

        print(
            f"{key:<30}: {value}"
        )
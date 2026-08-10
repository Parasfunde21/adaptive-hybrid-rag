import numpy as np


class RetrievalFeatureService:

    def extract(
        self,
        bm25_scores,
        dense_distances,
        bm25_docs,
        dense_docs
    ):

        bm25_scores = np.asarray(
            bm25_scores,
            dtype=float
        )

        dense_similarities = np.asarray(
            [
                1.0 / (1.0 + float(distance))
                for distance in dense_distances
            ],
            dtype=float
        )

        # -------------------------------------------------
        # BM25 statistics
        # -------------------------------------------------

        if len(bm25_scores) > 0:

            bm25_top = float(
                bm25_scores[0]
            )

            bm25_mean = float(
                np.mean(bm25_scores)
            )

            bm25_std = float(
                np.std(bm25_scores)
            )

            bm25_min = float(
                np.min(bm25_scores)
            )

            bm25_gap = (
                float(
                    bm25_scores[0]
                    -
                    bm25_scores[1]
                )
                if len(bm25_scores) > 1
                else 0.0
            )

        else:

            bm25_top = 0.0
            bm25_mean = 0.0
            bm25_std = 0.0
            bm25_min = 0.0
            bm25_gap = 0.0

        # -------------------------------------------------
        # Dense statistics
        # -------------------------------------------------

        if len(dense_similarities) > 0:

            dense_top = float(
                dense_similarities[0]
            )

            dense_mean = float(
                np.mean(
                    dense_similarities
                )
            )

            dense_std = float(
                np.std(
                    dense_similarities
                )
            )

            dense_min = float(
                np.min(
                    dense_similarities
                )
            )

            dense_gap = (
                float(
                    dense_similarities[0]
                    -
                    dense_similarities[1]
                )
                if len(dense_similarities) > 1
                else 0.0
            )

        else:

            dense_top = 0.0
            dense_mean = 0.0
            dense_std = 0.0
            dense_min = 0.0
            dense_gap = 0.0

        # -------------------------------------------------
        # Retrieval overlap
        # -------------------------------------------------

        bm25_set = set(
            bm25_docs
        )

        dense_set = set(
            dense_docs
        )

        union = (
            bm25_set
            |
            dense_set
        )

        intersection = (
            bm25_set
            &
            dense_set
        )

        overlap_ratio = (

            len(intersection)
            /
            max(len(union), 1)

        )

        # -------------------------------------------------
        # Rank agreement
        # -------------------------------------------------

        dense_rank = {
            doc: rank
            for rank, doc
            in enumerate(
                dense_docs,
                start=1
            )
        }

        bm25_rank = {
            doc: rank
            for rank, doc
            in enumerate(
                bm25_docs,
                start=1
            )
        }

        common_docs = (
            set(dense_rank)
            &
            set(bm25_rank)
        )

        if common_docs:

            differences = [

                abs(
                    dense_rank[doc]
                    -
                    bm25_rank[doc]
                )

                for doc
                in common_docs

            ]

            max_rank = max(
                len(dense_docs),
                len(bm25_docs),
                1
            )

            rank_agreement = (

                1.0
                -
                (
                    np.mean(differences)
                    /
                    max_rank
                )

            )

            rank_agreement = float(
                np.clip(
                    rank_agreement,
                    0.0,
                    1.0
                )
            )

        else:

            rank_agreement = 0.0

        return {

            "bm25_top_score":
                bm25_top,

            "bm25_mean_score":
                bm25_mean,

            "bm25_std_score":
                bm25_std,

            "bm25_min_score":
                bm25_min,

            "bm25_score_gap":
                bm25_gap,

            "dense_top_similarity":
                dense_top,

            "dense_mean_similarity":
                dense_mean,

            "dense_std_similarity":
                dense_std,

            "dense_min_similarity":
                dense_min,

            "dense_similarity_gap":
                dense_gap,

            "retrieval_overlap_ratio":
                overlap_ratio,

            "retrieval_rank_agreement":
                rank_agreement,

            "bm25_result_count":
                len(bm25_docs),

            "dense_result_count":
                len(dense_docs)
        }


retrieval_feature_service = (
    RetrievalFeatureService()
)
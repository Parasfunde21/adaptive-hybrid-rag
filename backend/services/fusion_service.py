import numpy as np


class AdaptiveScoreFusion:

    """
    Adaptive Score Fusion (ASF)

    Combines normalized BM25 scores and dense similarities
    using ML-predicted weights while preserving document
    metadata for downstream citation support.
    """

    # ========================================================
    # Normalization
    # ========================================================

    def normalize(self, scores):

        scores = np.array(
            scores,
            dtype=float
        )

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

    # ========================================================
    # Dense Distance -> Similarity
    # ========================================================

    def dense_similarity(self, distances):

        return np.array([
            1 / (1 + d)
            for d in distances
        ])

    # ========================================================
    # Fusion
    # ========================================================

    def fuse(
        self,
        dense_docs,
        dense_distances,
        dense_ids,
        dense_metadatas,
        bm25_docs,
        bm25_scores,
        bm25_ids=None,
        bm25_metadatas=None,
        dense_weight=0.5,
        bm25_weight=0.5
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

        # ====================================================
        # Dense Retrieval
        # ====================================================

        for rank, (
            doc,
            similarity,
            doc_id,
            metadata
        ) in enumerate(

            zip(
                dense_docs,
                dense_scores,
                dense_ids,
                dense_metadatas
            ),

            start=1
        ):

            # Use stable chunk ID instead of document text
            key = (
                metadata.get(
                    "chunk_id"
                )
                if metadata
                else doc_id
            )

            if key not in results:

                results[key] = {

                    "document":
                        doc,

                    "id":
                        doc_id,

                    "metadata":
                        metadata or {},

                    "fusion_score":
                        0.0,

                    "retrieved_by":
                        [],

                    "dense_similarity":
                        None,

                    "bm25_score":
                        None,

                    "dense_rank":
                        None,

                    "bm25_rank":
                        None
                }

            results[key][
                "fusion_score"
            ] += (

                dense_weight *

                float(
                    similarity
                )
            )

            results[key][
                "dense_similarity"
            ] = float(
                similarity
            )

            results[key][
                "dense_rank"
            ] = rank

            if "Dense" not in results[key][
                "retrieved_by"
            ]:

                results[key][
                    "retrieved_by"
                ].append(
                    "Dense"
                )

        # ====================================================
        # BM25 Retrieval
        # ====================================================

        for rank, (
            doc,
            score
        ) in enumerate(

            zip(
                bm25_docs,
                bm25_scores
            ),

            start=1
        ):

            # Try to match BM25 result to an existing
            # dense result using exact document text.
            matching_key = None

            for key, item in results.items():

                if item["document"] == doc:

                    matching_key = key
                    break

            # If BM25 found a document that Dense did not,
            # create a new result.
            if matching_key is None:

                if bm25_metadatas and (
                    rank - 1
                ) < len(bm25_metadatas):

                    metadata = (
                        bm25_metadatas[
                            rank - 1
                        ]
                        or {}
                    )

                else:

                    metadata = {}

                if bm25_ids and (
                    rank - 1
                ) < len(bm25_ids):

                    doc_id = (
                        bm25_ids[
                            rank - 1
                        ]
                    )

                else:

                    doc_id = (
                        metadata.get(
                            "chunk_id",
                            f"bm25_{rank}"
                        )
                    )

                matching_key = (
                    metadata.get(
                        "chunk_id",
                        doc_id
                    )
                )

                results[matching_key] = {

                    "document":
                        doc,

                    "id":
                        doc_id,

                    "metadata":
                        metadata,

                    "fusion_score":
                        0.0,

                    "retrieved_by":
                        [],

                    "dense_similarity":
                        None,

                    "bm25_score":
                        None,

                    "dense_rank":
                        None,

                    "bm25_rank":
                        None
                }

            results[
                matching_key
            ][
                "fusion_score"
            ] += (

                bm25_weight *

                float(
                    score
                )
            )

            results[
                matching_key
            ][
                "bm25_score"
            ] = float(
                score
            )

            results[
                matching_key
            ][
                "bm25_rank"
            ] = rank

            if "BM25" not in results[
                matching_key
            ][
                "retrieved_by"
            ]:

                results[
                    matching_key
                ][
                    "retrieved_by"
                ].append(
                    "BM25"
                )

        # ====================================================
        # Sort
        # ====================================================

        ranked = sorted(

            results.values(),

            key=lambda x:
                x["fusion_score"],

            reverse=True
        )

        # ====================================================
        # Round scores
        # ====================================================

        for item in ranked:

            item[
                "fusion_score"
            ] = round(

                item[
                    "fusion_score"
                ],

                6
            )

            if item[
                "dense_similarity"
            ] is not None:

                item[
                    "dense_similarity"
                ] = round(

                    item[
                        "dense_similarity"
                    ],

                    6
                )

            if item[
                "bm25_score"
            ] is not None:

                item[
                    "bm25_score"
                ] = round(

                    item[
                        "bm25_score"
                    ],

                    6
                )

        return ranked


fusion_service = AdaptiveScoreFusion()
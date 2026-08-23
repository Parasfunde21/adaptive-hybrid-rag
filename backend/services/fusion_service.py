import numpy as np


class AdaptiveScoreFusion:
    """
    Adaptive Score Fusion (ASF)

    Combines BM25 lexical retrieval and dense semantic
    retrieval using query-specific weights.

    Design goals:
    - Normalize both retrieval signals independently.
    - Match documents using stable document IDs whenever possible.
    - Preserve retrieval metadata.
    - Support documents appearing in only one retriever.
    - Make adaptive weights directly influence ranking.
    - Avoid document-text matching where stable IDs exist.
    """

    # ============================================================
    # Normalization
    # ============================================================

    @staticmethod
    def min_max_normalize(scores):
        """
        Min-max normalize scores to [0, 1].

        Higher input score means higher normalized score.
        """

        scores = np.asarray(
            scores,
            dtype=float
        )

        if scores.size == 0:
            return scores

        minimum = np.min(scores)
        maximum = np.max(scores)

        if maximum == minimum:
            return np.ones_like(scores)

        return (
            scores - minimum
        ) / (
            maximum - minimum
        )

    # ============================================================
    # Dense distance -> similarity
    # ============================================================

    @staticmethod
    def distance_to_similarity(distances):
        """
        Convert Chroma distance values into a similarity signal.

        Lower distance = higher similarity.
        """

        distances = np.asarray(
            distances,
            dtype=float
        )

        if distances.size == 0:
            return distances

        return 1.0 / (1.0 + distances)

    # ============================================================
    # Internal result creation
    # ============================================================

    @staticmethod
    def _create_result(
        document,
        doc_id,
        metadata
    ):
        """
        Create a standard fusion result object.
        """

        return {
            "document": document,
            "id": doc_id,
            "metadata": metadata or {},

            "fusion_score": 0.0,

            "dense_component": 0.0,
            "bm25_component": 0.0,

            "dense_similarity": None,
            "bm25_score": None,

            "dense_rank": None,
            "bm25_rank": None,

            "retrieved_by": []
        }

    # ============================================================
    # Stable document key
    # ============================================================

    @staticmethod
    def _document_key(
        doc_id,
        metadata,
        document
    ):
        """
        Generate a stable key for joining retrieval results.

        Priority:
        1. chunk_id
        2. document_id
        3. retriever ID
        4. document text fallback
        """

        metadata = metadata or {}

        chunk_id = metadata.get(
            "chunk_id"
        )

        if chunk_id:
            return str(chunk_id)

        document_id = metadata.get(
            "document_id"
        )

        if document_id:
            return str(document_id)

        if doc_id:
            return str(doc_id)

        return f"text::{hash(document)}"

    # ============================================================
    # Fusion
    # ============================================================

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
        """
        Perform weighted hybrid fusion.

        Final score:

            score =
                dense_weight * dense_signal
                +
                bm25_weight * bm25_signal

        Both signals are independently normalized before
        weighting.

        Parameters
        ----------
        dense_weight:
            Weight assigned to dense retrieval.

        bm25_weight:
            Weight assigned to BM25 retrieval.
        """

        # --------------------------------------------------------
        # Validate weights
        # --------------------------------------------------------

        dense_weight = float(
            dense_weight
        )

        bm25_weight = float(
            bm25_weight
        )

        if dense_weight < 0:
            raise ValueError(
                "dense_weight cannot be negative."
            )

        if bm25_weight < 0:
            raise ValueError(
                "bm25_weight cannot be negative."
            )

        weight_sum = (
            dense_weight +
            bm25_weight
        )

        if weight_sum <= 0:
            raise ValueError(
                "At least one fusion weight must be > 0."
            )

        # Normalize weights so that:

        # dense_weight + bm25_weight = 1

        dense_weight /= weight_sum
        bm25_weight /= weight_sum

        # --------------------------------------------------------
        # Convert dense distance -> similarity
        # --------------------------------------------------------

        dense_similarity = (
            self.distance_to_similarity(
                dense_distances
            )
        )

        # --------------------------------------------------------
        # Normalize each retrieval signal
        # --------------------------------------------------------

        dense_normalized = (
            self.min_max_normalize(
                dense_similarity
            )
        )

        bm25_normalized = (
            self.min_max_normalize(
                bm25_scores
            )
        )

        # --------------------------------------------------------
        # Result storage
        # --------------------------------------------------------

        results = {}

        # ========================================================
        # Dense results
        # ========================================================

        for rank, (
            document,
            normalized_score,
            raw_similarity,
            doc_id,
            metadata
        ) in enumerate(

            zip(
                dense_docs,
                dense_normalized,
                dense_similarity,
                dense_ids,
                dense_metadatas
            ),

            start=1
        ):

            metadata = (
                metadata or {}
            )

            key = self._document_key(
                doc_id,
                metadata,
                document
            )

            if key not in results:

                results[key] = (
                    self._create_result(
                        document,
                        doc_id,
                        metadata
                    )
                )

            result = results[key]

            result["dense_component"] = (
                float(normalized_score)
            )

            result["dense_similarity"] = (
                float(raw_similarity)
            )

            result["dense_rank"] = rank

            if "Dense" not in result[
                "retrieved_by"
            ]:

                result[
                    "retrieved_by"
                ].append("Dense")

        # ========================================================
        # BM25 results
        # ========================================================

        for rank, (
            document,
            normalized_score,
            raw_score,
            doc_id,
            metadata
        ) in enumerate(

            zip(
                bm25_docs,
                bm25_normalized,
                bm25_scores,
                (
                    bm25_ids
                    if bm25_ids is not None
                    else [None] * len(bm25_docs)
                ),
                (
                    bm25_metadatas
                    if bm25_metadatas is not None
                    else [{}] * len(bm25_docs)
                )
            ),

            start=1
        ):

            metadata = (
                metadata or {}
            )

            key = self._document_key(
                doc_id,
                metadata,
                document
            )

            if key not in results:

                results[key] = (
                    self._create_result(
                        document,
                        doc_id,
                        metadata
                    )
                )

            result = results[key]

            result["bm25_component"] = (
                float(normalized_score)
            )

            result["bm25_score"] = (
                float(raw_score)
            )

            result["bm25_rank"] = rank

            if "BM25" not in result[
                "retrieved_by"
            ]:

                result[
                    "retrieved_by"
                ].append("BM25")

        # ========================================================
        # Calculate final adaptive fusion score
        # ========================================================

        for result in results.values():

            dense_component = float(
                result["dense_component"]
            )

            bm25_component = float(
                result["bm25_component"]
            )

            result["fusion_score"] = (

                dense_weight *
                dense_component

                +

                bm25_weight *
                bm25_component
            )

        # ========================================================
        # Sort
        # ========================================================

        ranked = sorted(

            results.values(),

            key=lambda result:
                result["fusion_score"],

            reverse=True
        )

        # ========================================================
        # Store final weights in every result
        # ========================================================

        for result in ranked:

            result["dense_weight"] = round(
                dense_weight,
                6
            )

            result["bm25_weight"] = round(
                bm25_weight,
                6
            )

            result["fusion_score"] = round(
                result["fusion_score"],
                6
            )

            if result[
                "dense_similarity"
            ] is not None:

                result[
                    "dense_similarity"
                ] = round(
                    result[
                        "dense_similarity"
                    ],
                    6
                )

            if result[
                "bm25_score"
            ] is not None:

                result[
                    "bm25_score"
                ] = round(
                    result[
                        "bm25_score"
                    ],
                    6
                )

            result[
                "dense_component"
            ] = round(
                result[
                    "dense_component"
                ],
                6
            )

            result[
                "bm25_component"
            ] = round(
                result[
                    "bm25_component"
                ],
                6
            )

        return ranked


# ================================================================
# Global Fusion Service
# ================================================================

fusion_service = AdaptiveScoreFusion()
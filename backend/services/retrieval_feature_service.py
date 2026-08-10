import re
import numpy as np


class RetrievalFeatureService:
    """
    Extract retrieval features used by the adaptive RAG pipeline.

    This implementation is intentionally backward compatible with:

    1. Legacy V3 positional call:
       extract(
           bm25_scores,
           dense_distances,
           bm25_docs,
           dense_docs
       )

    2. V4 positional call:
       extract(
           query,
           bm25_scores,
           dense_distances,
           bm25_docs,
           dense_docs
       )

    3. Keyword-based calls:
       extract(
           query=query,
           bm25_scores=...,
           dense_distances=...,
           bm25_docs=...,
           dense_docs=...
       )

    4. Mixed calls:
       extract(
           query,
           bm25_scores=...,
           dense_distances=...,
           bm25_docs=...,
           dense_docs=...
       )
    """

    # ============================================================
    # Tokenization
    # ============================================================

    @staticmethod
    def tokenize(text):

        if text is None:
            return []

        return re.findall(
            r"\b[\w+#]+\b",
            str(text).lower()
        )

    # ============================================================
    # Lexical Coverage
    # ============================================================

    @classmethod
    def lexical_coverage(
        cls,
        query,
        document
    ):

        query_tokens = set(
            cls.tokenize(query)
        )

        if not query_tokens:
            return 0.0

        document_tokens = set(
            cls.tokenize(document)
        )

        matched_tokens = (
            query_tokens
            &
            document_tokens
        )

        return float(
            len(matched_tokens)
            /
            len(query_tokens)
        )

    # ============================================================
    # Argument Resolution
    # ============================================================

    def _resolve_arguments(
        self,
        args,
        kwargs
    ):

        """
        Resolve positional, keyword and mixed calls.

        Supported forms:

        4 positional:
            bm25_scores,
            dense_distances,
            bm25_docs,
            dense_docs

        5 positional:
            query,
            bm25_scores,
            dense_distances,
            bm25_docs,
            dense_docs

        Keyword:
            query=...
            bm25_scores=...
            dense_distances=...
            bm25_docs=...
            dense_docs=...

        Mixed:
            query,
            bm25_scores=...
            dense_distances=...
            bm25_docs=...
            dense_docs=...
        """

        kwargs = dict(kwargs)

        query = kwargs.pop(
            "query",
            None
        )

        bm25_scores = kwargs.pop(
            "bm25_scores",
            None
        )

        dense_distances = kwargs.pop(
            "dense_distances",
            None
        )

        bm25_docs = kwargs.pop(
            "bm25_docs",
            None
        )

        dense_docs = kwargs.pop(
            "dense_docs",
            None
        )

        # --------------------------------------------------------
        # Reject unknown keyword arguments
        # --------------------------------------------------------

        if kwargs:

            raise TypeError(
                "Unexpected keyword arguments: "
                f"{list(kwargs.keys())}"
            )

        # --------------------------------------------------------
        # Resolve positional arguments
        # --------------------------------------------------------

        if len(args) == 5:

            if query is not None:
                raise TypeError(
                    "query was provided both positionally "
                    "and as a keyword."
                )

            query = args[0]

            if bm25_scores is not None:
                raise TypeError(
                    "bm25_scores was provided both "
                    "positionally and as a keyword."
                )

            if dense_distances is not None:
                raise TypeError(
                    "dense_distances was provided both "
                    "positionally and as a keyword."
                )

            if bm25_docs is not None:
                raise TypeError(
                    "bm25_docs was provided both "
                    "positionally and as a keyword."
                )

            if dense_docs is not None:
                raise TypeError(
                    "dense_docs was provided both "
                    "positionally and as a keyword."
                )

            (
                query,
                bm25_scores,
                dense_distances,
                bm25_docs,
                dense_docs
            ) = args

        elif len(args) == 4:

            # Legacy V3:
            #
            # extract(
            #     bm25_scores,
            #     dense_distances,
            #     bm25_docs,
            #     dense_docs
            # )

            if query is not None:

                raise TypeError(
                    "Ambiguous 4 positional arguments with "
                    "query keyword."
                )

            if bm25_scores is not None:
                raise TypeError(
                    "bm25_scores was provided both "
                    "positionally and as a keyword."
                )

            if dense_distances is not None:
                raise TypeError(
                    "dense_distances was provided both "
                    "positionally and as a keyword."
                )

            if bm25_docs is not None:
                raise TypeError(
                    "bm25_docs was provided both "
                    "positionally and as a keyword."
                )

            if dense_docs is not None:
                raise TypeError(
                    "dense_docs was provided both "
                    "positionally and as a keyword."
                )

            (
                bm25_scores,
                dense_distances,
                bm25_docs,
                dense_docs
            ) = args

            query = ""

        elif len(args) == 1:

            # Mixed call:
            #
            # extract(
            #     query,
            #     bm25_scores=...,
            #     dense_distances=...,
            #     bm25_docs=...,
            #     dense_docs=...
            # )

            if query is not None:

                raise TypeError(
                    "query was provided both positionally "
                    "and as a keyword."
                )

            query = args[0]

        elif len(args) == 0:

            # Pure keyword call.
            pass

        else:

            raise TypeError(
                "Invalid positional arguments for "
                "RetrievalFeatureService.extract(). "
                f"Received {len(args)} positional arguments."
            )

        # --------------------------------------------------------
        # Defaults
        # --------------------------------------------------------

        if query is None:
            query = ""

        if bm25_scores is None:
            bm25_scores = []

        if dense_distances is None:
            dense_distances = []

        if bm25_docs is None:
            bm25_docs = []

        if dense_docs is None:
            dense_docs = []

        return (
            query,
            bm25_scores,
            dense_distances,
            bm25_docs,
            dense_docs
        )

    # ============================================================
    # Main Feature Extraction
    # ============================================================

    def extract(
        self,
        *args,
        **kwargs
    ):

        (
            query,
            bm25_scores,
            dense_distances,
            bm25_docs,
            dense_docs
        ) = self._resolve_arguments(
            args,
            kwargs
        )

        # --------------------------------------------------------
        # Convert inputs safely
        # --------------------------------------------------------

        bm25_scores = np.asarray(
            bm25_scores,
            dtype=float
        )

        dense_distances = np.asarray(
            dense_distances,
            dtype=float
        )

        bm25_docs = list(
            bm25_docs
        )

        dense_docs = list(
            dense_docs
        )

        # ========================================================
        # BM25 Statistics
        # ========================================================

        if len(bm25_scores) > 0:

            bm25_top = float(
                bm25_scores[0]
            )

            bm25_mean = float(
                np.mean(
                    bm25_scores
                )
            )

            bm25_std = float(
                np.std(
                    bm25_scores
                )
            )

            bm25_min = float(
                np.min(
                    bm25_scores
                )
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

            bm25_top_mean_ratio = (

                float(
                    bm25_top
                    /
                    bm25_mean
                )

                if bm25_mean != 0

                else 0.0
            )

        else:

            bm25_top = 0.0
            bm25_mean = 0.0
            bm25_std = 0.0
            bm25_min = 0.0
            bm25_gap = 0.0
            bm25_top_mean_ratio = 0.0

        # ========================================================
        # Dense Similarities
        # ========================================================

        dense_similarities = np.asarray(
            [
                1.0
                /
                (
                    1.0
                    +
                    float(distance)
                )

                for distance
                in dense_distances
            ],
            dtype=float
        )

        # ========================================================
        # Dense Statistics
        # ========================================================

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

            dense_top_mean_ratio = (

                float(
                    dense_top
                    /
                    dense_mean
                )

                if dense_mean != 0

                else 0.0
            )

        else:

            dense_top = 0.0
            dense_mean = 0.0
            dense_std = 0.0
            dense_min = 0.0
            dense_gap = 0.0
            dense_top_mean_ratio = 0.0

        # ========================================================
        # Retrieval Overlap
        # ========================================================

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

        overlap_ratio = float(
            len(intersection)
            /
            max(
                len(union),
                1
            )
        )

        # ========================================================
        # Rank Agreement
        # ========================================================

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

            rank_agreement = float(
                np.clip(
                    1.0
                    -
                    (
                        np.mean(
                            differences
                        )
                        /
                        max_rank
                    ),
                    0.0,
                    1.0
                )
            )

        else:

            rank_agreement = 0.0

        # ========================================================
        # Query Features
        # ========================================================

        query_tokens = self.tokenize(
            query
        )

        query_token_count = len(
            query_tokens
        )

        query_unique_token_count = len(
            set(
                query_tokens
            )
        )

        query_avg_token_length = (

            float(
                np.mean(
                    [
                        len(token)
                        for token
                        in query_tokens
                    ]
                )
            )

            if query_tokens

            else 0.0
        )

        # ========================================================
        # Lexical Coverage
        # ========================================================

        if query:

            if bm25_docs:

                bm25_top_lexical_coverage = (
                    self.lexical_coverage(
                        query,
                        bm25_docs[0]
                    )
                )

            else:

                bm25_top_lexical_coverage = 0.0

            if dense_docs:

                dense_top_lexical_coverage = (
                    self.lexical_coverage(
                        query,
                        dense_docs[0]
                    )
                )

            else:

                dense_top_lexical_coverage = 0.0

        else:

            bm25_top_lexical_coverage = 0.0
            dense_top_lexical_coverage = 0.0

        # ========================================================
        # Return Features
        # ========================================================

        return {

            # ----------------------------------------------------
            # V4 Rich features
            # ----------------------------------------------------

            "bm25_top_score":
                bm25_top,

            "bm25_mean_score":
                bm25_mean,

            "bm25_score_std":
                bm25_std,

            "bm25_score_gap":
                bm25_gap,

            "bm25_top_mean_ratio":
                bm25_top_mean_ratio,

            "dense_top_similarity":
                dense_top,

            "dense_mean_similarity":
                dense_mean,

            "dense_similarity_std":
                dense_std,

            "dense_similarity_gap":
                dense_gap,

            "dense_top_mean_ratio":
                dense_top_mean_ratio,

            "overlap_ratio":
                overlap_ratio,

            "rank_agreement":
                rank_agreement,

            "query_token_count":
                query_token_count,

            "query_unique_token_count":
                query_unique_token_count,

            "query_avg_token_length":
                query_avg_token_length,

            "bm25_top_lexical_coverage":
                bm25_top_lexical_coverage,

            "dense_top_lexical_coverage":
                dense_top_lexical_coverage,

            # ----------------------------------------------------
            # Legacy V3 compatibility
            # ----------------------------------------------------

            "bm25_std_score":
                bm25_std,

            "bm25_min_score":
                bm25_min,

            "dense_std_similarity":
                dense_std,

            "dense_min_similarity":
                dense_min,

            "retrieval_overlap_ratio":
                overlap_ratio,

            "retrieval_rank_agreement":
                rank_agreement,

            "bm25_result_count":
                len(bm25_docs),

            "dense_result_count":
                len(dense_docs)
        }


# ============================================================
# Global Service Instance
# ============================================================

retrieval_feature_service = (
    RetrievalFeatureService()
)
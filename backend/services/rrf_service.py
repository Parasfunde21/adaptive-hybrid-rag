class ReciprocalRankFusion:

    """
    Standard Reciprocal Rank Fusion (RRF)

    Score = Σ 1 / (k + rank)

    Reference:
    Cormack et al. (2009)
    """

    def __init__(self):

        self.k = 60

    def fuse(

        self,

        dense_docs,

        bm25_docs

    ):

        scores = {}

        # -------------------------
        # Dense Contribution
        # -------------------------

        for rank, doc in enumerate(

            dense_docs,

            start=1

        ):

            if doc not in scores:

                scores[doc] = {

                    "document": doc,

                    "score": 0.0,

                    "source": []

                }

            scores[doc]["score"] += (

                1 /

                (self.k + rank)

            )

            scores[doc]["source"].append(

                "Dense"

            )

        # -------------------------
        # BM25 Contribution
        # -------------------------

        for rank, doc in enumerate(

            bm25_docs,

            start=1

        ):

            if doc not in scores:

                scores[doc] = {

                    "document": doc,

                    "score": 0.0,

                    "source": []

                }

            scores[doc]["score"] += (

                1 /

                (self.k + rank)

            )

            scores[doc]["source"].append(

                "BM25"

            )

        ranked = sorted(

            scores.values(),

            key=lambda x: x["score"],

            reverse=True

        )

        return ranked


rrf = ReciprocalRankFusion()
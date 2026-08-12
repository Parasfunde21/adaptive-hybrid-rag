import math
import re
from collections import Counter

from services.document_loader import (
    get_all_documents
)


# ============================================================
# Corpus Statistics
# ============================================================

class CorpusStatistics:
    """
    Computes corpus-level lexical statistics used by
    the adaptive retrieval pipeline.

    The statistics are built from the complete indexed
    corpus returned by the batch-safe document loader.
    """

    # --------------------------------------------------------
    # Initialization
    # --------------------------------------------------------

    def __init__(self):

        print(
            "\n"
            + "=" * 70
        )

        print(
            "INITIALIZING CORPUS STATISTICS"
        )

        print(
            "=" * 70
        )

        self.documents = (
            get_all_documents()
        )

        self.total_documents = (
            len(self.documents)
        )

        self.document_frequency = (
            Counter()
        )

        self.term_frequency = (
            Counter()
        )

        print(
            "Building corpus statistics..."
        )

        self.build()

        print(
            "Corpus statistics ready."
        )

        print(
            f"Total documents : "
            f"{self.total_documents}"
        )

        print(
            f"Unique terms    : "
            f"{len(self.document_frequency)}"
        )

        print(
            "=" * 70
        )

    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    def tokenize(
        self,
        text
    ):
        """
        Tokenize a document or query.

        Keeps alphanumeric terms as well as
        common programming-related symbols such
        as # and +.
        """

        if not text:

            return []

        return re.findall(
            r"\b[\w#+]+\b",
            str(text).lower()
        )

    # --------------------------------------------------------
    # Build Statistics
    # --------------------------------------------------------

    def build(self):
        """
        Build:

        1. Document frequency
        2. Term frequency

        Document frequency counts the number of
        documents containing a term.

        Term frequency counts the total occurrences
        of a term across the corpus.
        """

        total = (
            self.total_documents
        )

        if total == 0:

            return

        for index, document in enumerate(
            self.documents,
            start=1
        ):

            tokens = self.tokenize(
                document
            )

            if not tokens:

                continue

            # ------------------------------------------------
            # Document Frequency
            # ------------------------------------------------

            unique_tokens = set(
                tokens
            )

            for token in unique_tokens:

                self.document_frequency[
                    token
                ] += 1

            # ------------------------------------------------
            # Term Frequency
            # ------------------------------------------------

            for token in tokens:

                self.term_frequency[
                    token
                ] += 1

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                index == 1
                or index % 5000 == 0
                or index == total
            ):

                percentage = (
                    index / total
                ) * 100

                print(
                    f"Statistics: "
                    f"{index:>6}/{total} "
                    f"({percentage:6.2f}%)"
                )

    # --------------------------------------------------------
    # Document Frequency
    # --------------------------------------------------------

    def df(
        self,
        token
    ):
        """
        Return document frequency for a term.
        """

        if not token:

            return 0

        return self.document_frequency.get(
            str(token).lower(),
            0
        )

    # --------------------------------------------------------
    # Term Frequency
    # --------------------------------------------------------

    def tf(
        self,
        token
    ):
        """
        Return corpus-wide term frequency.
        """

        if not token:

            return 0

        return self.term_frequency.get(
            str(token).lower(),
            0
        )

    # --------------------------------------------------------
    # Inverse Document Frequency
    # --------------------------------------------------------

    def idf(
        self,
        token
    ):
        """
        Compute smoothed IDF.

        Formula:

            IDF(t) =
                log((N + 1) / (DF(t) + 1)) + 1

        where:

            N  = total number of documents
            DF = number of documents containing term
        """

        df = self.df(
            token
        )

        return math.log(
            (
                self.total_documents
                + 1
            )
            /
            (
                df
                + 1
            )
        ) + 1


# ============================================================
# Global Corpus Statistics Instance
# ============================================================

corpus_stats = (
    CorpusStatistics()
)


# ============================================================
# Main Test
# ============================================================

if __name__ == "__main__":

    print(
        "\nCorpus statistics interactive test."
    )

    print(
        "Type 'exit' to stop."
    )

    while True:

        query = input(
            "\nTerm : "
        ).strip()

        if query.lower() == "exit":

            break

        print(
            "TF :",
            corpus_stats.tf(query)
        )

        print(
            "DF :",
            corpus_stats.df(query)
        )

        print(
            "IDF:",
            round(
                corpus_stats.idf(query),
                4
            )
        )
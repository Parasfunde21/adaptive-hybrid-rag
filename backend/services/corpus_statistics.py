import math
import re
from collections import Counter

from services.document_loader import get_all_documents


class CorpusStatistics:

    def __init__(self):

        self.documents = get_all_documents()

        self.total_documents = len(self.documents)

        self.document_frequency = Counter()

        self.term_frequency = Counter()

        self.build()

    def tokenize(self, text):

        return re.findall(
            r"\b[\w#+]+\b",
            text.lower()
        )

    def build(self):

        for document in self.documents:

            tokens = self.tokenize(document)

            unique_tokens = set(tokens)

            for token in unique_tokens:

                self.document_frequency[token] += 1

            for token in tokens:

                self.term_frequency[token] += 1

    def df(self, token):

        return self.document_frequency.get(
            token.lower(),
            0
        )

    def tf(self, token):

        return self.term_frequency.get(
            token.lower(),
            0
        )

    def idf(self, token):

        df = self.df(token)

        return math.log(
            (self.total_documents + 1) /
            (df + 1)
        ) + 1


corpus_stats = CorpusStatistics()


if __name__ == "__main__":

    while True:

        query = input("\nTerm : ")

        if query == "exit":
            break

        print("TF :", corpus_stats.tf(query))
        print("DF :", corpus_stats.df(query))
        print("IDF:", round(corpus_stats.idf(query), 4))
import re
import numpy as np

from services.corpus_statistics import corpus_stats


class FeatureExtractor:

    def tokenize(self, text):

        return re.findall(
            r"\b[\w\+\#]+\b",
            text.lower()
        )

    def extract(self, query):

        tokens = self.tokenize(query)

        if len(tokens) == 0:
            return {}

        idf_scores = [
            corpus_stats.idf(word)
            for word in tokens
        ]

        df_scores = [
            corpus_stats.df(word)
            for word in tokens
        ]

        tf_scores = [
            corpus_stats.tf(word)
            for word in tokens
        ]

        unknown_terms = sum(
            1
            for word in tokens
            if corpus_stats.df(word) == 0
        )

        return {

            "query_length":
                len(tokens),

            "avg_word_length":
                round(
                    np.mean(
                        [len(word) for word in tokens]
                    ),
                    3
                ),

            "average_idf":
                round(
                    np.mean(idf_scores),
                    3
                ),

            "maximum_idf":
                round(
                    np.max(idf_scores),
                    3
                ),

            "minimum_idf":
                round(
                    np.min(idf_scores),
                    3
                ),

            "average_df":
                round(
                    np.mean(df_scores),
                    3
                ),

            "maximum_df":
                int(
                    np.max(df_scores)
                ),

            "average_tf":
                round(
                    np.mean(tf_scores),
                    3
                ),

            "maximum_tf":
                int(
                    np.max(tf_scores)
                ),

            "oov_ratio":
                round(
                    unknown_terms / len(tokens),
                    3
                ),

            "digit_ratio":
                round(
                    sum(
                        word.isdigit()
                        for word in tokens
                    ) / len(tokens),
                    3
                ),

            "uppercase_ratio":
                round(
                    sum(
                        word.isupper()
                        for word in query.split()
                    ) / max(len(query.split()), 1),
                    3
                ),

            "special_character_count":
                len(
                    re.findall(
                        r"[/:._\-]",
                        query
                    )
                )
        }


feature_extractor = FeatureExtractor()


if __name__ == "__main__":

    while True:

        query = input("\nQuery : ")

        if query.lower() == "exit":
            break

        features = feature_extractor.extract(query)

        print()

        for key, value in features.items():

            print(f"{key:28}: {value}")
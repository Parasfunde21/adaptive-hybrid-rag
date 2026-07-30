import re


class QueryAnalyzer:
    """
    Rule-based Query Analyzer

    Classifies a query into:
    - conceptual
    - keyword
    - mixed

    and assigns adaptive retrieval weights.
    """

    def __init__(self):

        self.conceptual_words = {
            "explain",
            "describe",
            "difference",
            "compare",
            "advantages",
            "disadvantages",
            "benefits",
            "overview",
            "working",
            "architecture",
            "concept",
            "why",
            "how"
        }

        self.keyword_patterns = [
            r"\b\d+\b",          # Numbers
            r"http",
            r"https",
            r"404",
            r"500",
            r"tcp",
            r"udp",
            r"sql",
            r"html",
            r"css",
            r"json",
            r"xml",
            r"api",
            r"rest",
            r"jwt",
            r"oauth"
        ]

    def analyze(self, query: str):

        query_lower = query.lower()

        conceptual_score = 0
        keyword_score = 0

        # Check conceptual words
        for word in self.conceptual_words:
            if word in query_lower:
                conceptual_score += 1

        # Check technical patterns
        for pattern in self.keyword_patterns:
            if re.search(pattern, query_lower):
                keyword_score += 1

        # Classification
        if conceptual_score > keyword_score:

            return {
                "query_type": "conceptual",
                "dense_weight": 0.8,
                "bm25_weight": 0.2
            }

        elif keyword_score > conceptual_score:

            return {
                "query_type": "keyword",
                "dense_weight": 0.2,
                "bm25_weight": 0.8
            }

        else:

            return {
                "query_type": "mixed",
                "dense_weight": 0.5,
                "bm25_weight": 0.5
            }


# Global instance
query_analyzer = QueryAnalyzer()


def analyze_query(query: str):
    return query_analyzer.analyze(query)


if __name__ == "__main__":

    while True:

        query = input("\nEnter Query : ")

        if query.lower() == "exit":
            break

        result = analyze_query(query)

        print("\nAnalysis")
        print("-" * 50)

        for key, value in result.items():
            print(f"{key:15}: {value}")
from keybert import KeyBERT

from services.document_loader import get_all_documents

kw_model = KeyBERT("all-MiniLM-L6-v2")


class QueryGenerator:

    def __init__(self):
        self.documents = get_all_documents()

    def generate_queries(self):

        generated_queries = []

        for document in self.documents:

            keywords = kw_model.extract_keywords(
                document,
                keyphrase_ngram_range=(1, 3),
                stop_words="english",
                top_n=5
            )

            for keyword, score in keywords:

                generated_queries.append({
                    "query": keyword,
                    "source_document": document
                })

        return generated_queries


generator = QueryGenerator()

if __name__ == "__main__":

    queries = generator.generate_queries()

    print(f"Generated {len(queries)} queries\n")

    for item in queries[:20]:
        print(item["query"])
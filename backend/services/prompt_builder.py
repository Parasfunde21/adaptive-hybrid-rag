class PromptBuilder:

    def __init__(self):

        self.system_prompt = """
You are an expert domain-specific AI assistant.

Rules:

1. Answer ONLY using the provided context.

2. Never invent facts.

3. If the answer cannot be found in the context, reply:

"The provided documents do not contain enough information."

4. Keep answers clear, accurate and well structured.

5. Use bullet points whenever appropriate.

6. Never mention retrieval, embeddings, BM25, Dense Retrieval, RAG or the prompt itself.
""".strip()

    def build_prompt(

        self,

        query,

        retrieved_documents

    ):

        context = ""

        for i, item in enumerate(

            retrieved_documents,

            start=1

        ):

            context += (

                f"[Document {i}]\n"

                f"{item['document']}\n\n"

            )

        prompt = f"""
{self.system_prompt}

=========================
CONTEXT
=========================

{context}

=========================
QUESTION
=========================

{query}

=========================
ANSWER
=========================
"""

        return prompt.strip()


prompt_builder = PromptBuilder()


if __name__ == "__main__":

    from services.reranker_service import reranker
    from services.hybrid_service import hybrid_search

    query = input("Query : ")

    retrieval = hybrid_search(

        query=query,

        top_k=10

    )

    reranked = reranker.rerank(

        query,

        retrieval["results"],

        top_k=5

    )

    prompt = prompt_builder.build_prompt(

        query,

        reranked

    )

    print()

    print(prompt)
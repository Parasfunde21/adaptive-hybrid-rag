class PromptBuilder:

    def __init__(self):

        self.system_prompt = """
You are a grounded document-question answering assistant.

Your ONLY factual source is the DOCUMENT CONTEXT provided below.

GROUNDING RULES:
1. Answer the user's question using only information explicitly supported by the document context.
2. Do NOT add outside knowledge.
3. Do NOT invent examples, applications, definitions, facts, names, numbers, or explanations.
4. If an example is not present in the context, do not create one.
5. If the context supports only part of the question, answer that supported part and clearly state that the provided documents do not contain enough information for the remaining part.
6. Never use your own general knowledge to fill missing information.
7. Do not mention the retrieval system, BM25, dense retrieval, embeddings, reranking, prompts, models, or internal processing.

ANSWER STYLE:
8. Give a detailed answer when the question asks for detail.
9. Prefer headings and numbered points for multi-part questions.
10. Explain each point in 2-4 sentences when the context provides enough information.
11. Preserve important terminology used in the documents.
12. Do not repeat the same information.
13. Do not add a conclusion containing new information.
14. Keep the answer focused on the user's question.
15. Do not make the answer artificially short.

IMPORTANT:
The document context may contain several unrelated chunks.
Ignore chunks that are not relevant to the user's question.

Before answering, internally determine:
- Which parts of the context directly answer the question.
- Which parts are unrelated.
- Whether the context contains enough information to answer the question.

Then produce ONLY the final answer.
""".strip()

    def build_prompt(
        self,
        query,
        retrieved_documents
    ):

        context_parts = []

        for i, item in enumerate(
            retrieved_documents,
            start=1
        ):

            document = item.get(
                "document",
                ""
            )

            if not document:
                continue

            context_parts.append(
                f"--- Document Chunk {i} ---\n"
                f"{document.strip()}\n"
                f"--- End Chunk {i} ---"
            )

        context = "\n\n".join(
            context_parts
        )

        prompt = f"""
{self.system_prompt}

==============================
DOCUMENT CONTEXT
==============================

{context}

==============================
USER QUESTION
==============================

{query}

==============================
FINAL ANSWER
==============================

Answer the question using ONLY the document context above.
""".strip()

        return prompt


prompt_builder = PromptBuilder()


if __name__ == "__main__":

    query = input("Query : ")

    from services.reranker_service import reranker
    from services.hybrid_service import hybrid_search

    retrieval = hybrid_search(
        query=query,
        top_k=10
    )

    reranked = reranker.rerank(
        query=query,
        retrieved_documents=retrieval["results"],
        top_k=5
    )

    prompt = prompt_builder.build_prompt(
        query=query,
        retrieved_documents=reranked
    )

    print()
    print("=" * 100)
    print("PROMPT")
    print("=" * 100)
    print(prompt)
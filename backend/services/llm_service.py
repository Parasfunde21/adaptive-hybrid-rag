import ollama


class LLMService:

    def __init__(self):

        self.model = "qwen2.5:7b"

    def generate(

        self,

        prompt: str

    ):

        response = ollama.chat(

            model=self.model,

            messages=[

                {

                    "role": "user",

                    "content": prompt

                }

            ]

        )

        return response["message"]["content"]


llm_service = LLMService()


if __name__ == "__main__":

    from services.prompt_builder import prompt_builder
    from services.reranker_service import reranker
    from services.hybrid_service import hybrid_search

    while True:

        query = input("\nQuery : ")

        if query.lower() == "exit":
            break

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

        print("\nGenerating answer...\n")

        answer = llm_service.generate(prompt)

        print("=" * 100)
        print(answer)
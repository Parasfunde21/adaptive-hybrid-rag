from services.hybrid_service import hybrid_search
from services.reranker_service import reranker
from services.prompt_builder import prompt_builder
from services.llm_service import llm_service


class AdaptiveHybridRAG:

    def __init__(self):
        pass

    def answer(

        self,

        query,

        retrieval_top_k=10,

        rerank_top_k=5

    ):

        # ----------------------------------------
        # Step 1: Adaptive Hybrid Retrieval
        # ----------------------------------------

        retrieval = hybrid_search(

            query=query,

            top_k=retrieval_top_k

        )

        # ----------------------------------------
        # Step 2: Cross Encoder Reranking
        # ----------------------------------------

        reranked = reranker.rerank(

            query=query,

            retrieved_documents=retrieval["results"],

            top_k=rerank_top_k

        )

        # ----------------------------------------
        # Step 3: Prompt Construction
        # ----------------------------------------

        prompt = prompt_builder.build_prompt(

            query=query,

            retrieved_documents=reranked

        )

        # ----------------------------------------
        # Step 4: LLM Generation
        # ----------------------------------------

        answer = llm_service.generate(

            prompt

        )

        # ----------------------------------------
        # Step 5: Return Everything
        # ----------------------------------------

        return {

            "query": query,

            "weights": retrieval["weights"],

            "documents": reranked,

            "prompt": prompt,

            "answer": answer

        }


rag_pipeline = AdaptiveHybridRAG()


if __name__ == "__main__":

    while True:

        query = input("\nQuery : ")

        if query.lower() == "exit":
            break

        response = rag_pipeline.answer(query)

        print("\n" + "=" * 100)
        print("FINAL ANSWER")
        print("=" * 100)

        print(response["answer"])

        print("\n" + "=" * 100)
        print("ADAPTIVE WEIGHTS")
        print("=" * 100)

        print(response["weights"])

        print("\n" + "=" * 100)
        print("TOP DOCUMENTS")
        print("=" * 100)

        for i, doc in enumerate(response["documents"], start=1):

            print(f"\nRank {i}")

            print(f"Cross Score : {doc['cross_score']:.4f}")

            print(f"Fusion Score: {doc['fusion_score']:.4f}")

            print(f"Retrieved By: {', '.join(doc['retrieved_by'])}")

            print("-" * 100)

            print(doc["document"][:300])
import time

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

        total_start = time.perf_counter()

        # ====================================================
        # Step 1: Adaptive Hybrid Retrieval
        # ====================================================

        start = time.perf_counter()

        retrieval = hybrid_search(
            query=query,
            top_k=retrieval_top_k
        )

        retrieval_latency = (
            time.perf_counter() - start
        )

        # ====================================================
        # Step 2: Cross Encoder Reranking
        # ====================================================

        start = time.perf_counter()

        reranked = reranker.rerank(
            query=query,
            retrieved_documents=retrieval["results"],
            top_k=rerank_top_k
        )

        reranking_latency = (
            time.perf_counter() - start
        )

        # ====================================================
        # Step 3: Prompt Construction
        # ====================================================

        start = time.perf_counter()

        prompt = prompt_builder.build_prompt(
            query=query,
            retrieved_documents=reranked
        )

        prompt_latency = (
            time.perf_counter() - start
        )

        # ====================================================
        # Step 4: LLM Generation
        # ====================================================

        start = time.perf_counter()

        answer = llm_service.generate(
            prompt
        )

        generation_latency = (
            time.perf_counter() - start
        )

        # ====================================================
        # Total Latency
        # ====================================================

        total_latency = (
            time.perf_counter() - total_start
        )

        # ====================================================
        # Return Public API Data
        # ====================================================

        return {

            "query": query,

            "weights": retrieval["weights"],

            "documents": reranked,

            "answer": answer,

            "latency": {

                "retrieval_ms": round(
                    retrieval_latency * 1000,
                    2
                ),

                "reranking_ms": round(
                    reranking_latency * 1000,
                    2
                ),

                "prompt_ms": round(
                    prompt_latency * 1000,
                    2
                ),

                "generation_ms": round(
                    generation_latency * 1000,
                    2
                ),

                "total_ms": round(
                    total_latency * 1000,
                    2
                )

            }

        }


# ============================================================
# Singleton
# ============================================================

rag_pipeline = AdaptiveHybridRAG()


# ============================================================
# CLI Test
# ============================================================

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
        print("LATENCY")
        print("=" * 100)

        for key, value in response["latency"].items():

            print(
                f"{key:20}: {value:.2f} ms"
            )

        print("\n" + "=" * 100)
        print("TOP DOCUMENTS")
        print("=" * 100)

        for i, doc in enumerate(
            response["documents"],
            start=1
        ):

            print(f"\nRank {i}")

            print(
                f"Cross Score : "
                f"{doc.get('cross_score', 0):.4f}"
            )

            print(
                f"Fusion Score: "
                f"{doc.get('fusion_score', 0):.4f}"
            )

            print(
                f"Retrieved By: "
                f"{', '.join(doc.get('retrieved_by', []))}"
            )

            print(
                f"Source      : "
                f"{doc.get('source')}"
            )

            print(
                f"Chunk ID    : "
                f"{doc.get('chunk_id')}"
            )

            print("-" * 100)

            print(
                doc["document"][:300]
            )
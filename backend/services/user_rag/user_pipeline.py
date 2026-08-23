import time

from services.user_rag.user_retriever import (
    user_adaptive_retriever,
)

from services.reranker_service import (
    reranker,
)

from services.prompt_builder import (
    prompt_builder,
)

from services.llm_service import (
    llm_service,
)


class UserRAGPipeline:

    """
    User-specific Adaptive Hybrid RAG pipeline.

    Pipeline:

        User Query
             |
             v
        BM25 + Dense
             |
             v
        V4 Adaptive Weighting
             |
             v
        Adaptive Fusion
             |
             v
        Cross Encoder Reranking
             |
             v
        Context Expansion
             |
             v
        Prompt Construction
             |
             v
        Qwen2.5:7B
             |
             v
        Grounded Answer
    """

    def _expand_context(
        self,
        user_id,
        reranked_documents,
        max_chunks=8,
        window=2,
    ):

        """
        Expand highly relevant chunks with neighboring chunks
        from the same uploaded document.

        This prevents the LLM from receiving isolated chunks such as:

            chunk 2 -> Objectives 1 and 2

        while missing:

            chunk 3 -> Objectives 3 and 4
            chunk 4 -> Objectives 5 and 6

        The expansion is based on document_id/file_id and
        chunk_index metadata.
        """

        if not reranked_documents:
            return []

        selected = []

        # ----------------------------------------------------
        # First: keep the strongest reranked chunks
        # ----------------------------------------------------

        for item in reranked_documents:

            metadata = item.get(
                "metadata",
                {}
            ) or {}

            document_id = (
                metadata.get("document_id")
                or metadata.get("file_id")
            )

            chunk_index = metadata.get(
                "chunk_index"
            )

            selected.append({
                "item": item,
                "document_id": document_id,
                "chunk_index": chunk_index,
            })

        # ----------------------------------------------------
        # Get user collection
        # ----------------------------------------------------

        try:

            collection = (
                user_adaptive_retriever.get_collection(
                    user_id
                )
            )

        except Exception:

            # If expansion fails, safely return reranked
            # results instead of breaking the pipeline.

            return reranked_documents[:max_chunks]

        # ----------------------------------------------------
        # Cache complete user collection
        # ----------------------------------------------------

        try:

            collection_data = collection.get(
                include=[
                    "documents",
                    "metadatas",
                ]
            )

        except Exception:

            return reranked_documents[:max_chunks]

        all_documents = collection_data.get(
            "documents",
            []
        )

        all_metadatas = collection_data.get(
            "metadatas",
            []
        )

        # ----------------------------------------------------
        # Build lookup:
        #
        # (document_id, chunk_index) -> chunk
        # ----------------------------------------------------

        chunk_lookup = {}

        for document, metadata in zip(
            all_documents,
            all_metadatas
        ):

            metadata = metadata or {}

            document_id = (
                metadata.get("document_id")
                or metadata.get("file_id")
            )

            chunk_index = metadata.get(
                "chunk_index"
            )

            if (
                document_id is None
                or chunk_index is None
            ):

                continue

            try:

                chunk_index = int(
                    chunk_index
                )

            except (
                TypeError,
                ValueError
            ):

                continue

            chunk_lookup[
                (
                    document_id,
                    chunk_index
                )
            ] = {
                "document": document,
                "metadata": metadata,
            }

        # ----------------------------------------------------
        # Add original reranked chunks first
        # ----------------------------------------------------

        expanded = []

        seen = set()

        for selected_item in selected:

            item = selected_item["item"]

            metadata = (
                item.get(
                    "metadata",
                    {}
                )
                or {}
            )

            document_id = (
                metadata.get("document_id")
                or metadata.get("file_id")
            )

            chunk_index = metadata.get(
                "chunk_index"
            )

            key = (
                document_id,
                chunk_index
            )

            if key not in seen:

                expanded.append(
                    item
                )

                seen.add(key)

        # ----------------------------------------------------
        # Add neighboring chunks
        # ----------------------------------------------------

        for selected_item in selected:

            document_id = (
                selected_item["document_id"]
            )

            chunk_index = (
                selected_item["chunk_index"]
            )

            if (
                document_id is None
                or chunk_index is None
            ):

                continue

            try:

                chunk_index = int(
                    chunk_index
                )

            except (
                TypeError,
                ValueError
            ):

                continue

            for offset in range(
                -window,
                window + 1
            ):

                neighbor_index = (
                    chunk_index
                    + offset
                )

                key = (
                    document_id,
                    neighbor_index
                )

                if key in seen:

                    continue

                neighbor = chunk_lookup.get(
                    key
                )

                if neighbor is None:

                    continue

                expanded_item = {
                    "document":
                        neighbor["document"],

                    "metadata":
                        neighbor["metadata"],

                    "source":
                        "context_expansion",

                    "expanded_from":
                        chunk_index,
                }

                expanded.append(
                    expanded_item
                )

                seen.add(key)

                if len(expanded) >= max_chunks:

                    break

            if len(expanded) >= max_chunks:

                break

        # ----------------------------------------------------
        # Sort expanded context by document + chunk order
        # ----------------------------------------------------
        #
        # This makes the context coherent instead of:
        #
        # chunk 52
        # chunk 2
        # chunk 79
        #
        # ----------------------------------------------------

        def sort_key(item):

            metadata = (
                item.get(
                    "metadata",
                    {}
                )
                or {}
            )

            document_id = (
                metadata.get("document_id")
                or metadata.get("file_id")
                or ""
            )

            chunk_index = metadata.get(
                "chunk_index",
                999999
            )

            try:

                chunk_index = int(
                    chunk_index
                )

            except (
                TypeError,
                ValueError
            ):

                chunk_index = 999999

            return (
                document_id,
                chunk_index
            )

        expanded.sort(
            key=sort_key
        )

        return expanded[:max_chunks]

    # ========================================================
    # ANSWER
    # ========================================================

    def answer(
        self,
        user_id,
        query,
        retrieval_candidate_k=50,
        rerank_top_k=5,
        context_max_chunks=8,
        context_window=2,
    ):

        total_start = time.perf_counter()

        # ====================================================
        # 1. Adaptive Retrieval
        # ====================================================

        retrieval_start = time.perf_counter()

        retrieval = (
            user_adaptive_retriever.search(
                user_id=user_id,
                query=query,
                top_k=retrieval_candidate_k,
            )
        )

        retrieval_latency = (
            time.perf_counter()
            - retrieval_start
        )

        candidates = retrieval.get(
            "results",
            []
        )

        # ====================================================
        # 2. Cross Encoder Reranking
        # ====================================================

        reranking_start = time.perf_counter()

        reranked = reranker.rerank(
            query=query,
            retrieved_documents=candidates,
            top_k=rerank_top_k,
        )

        reranking_latency = (
            time.perf_counter()
            - reranking_start
        )

        # ====================================================
        # 3. Context Expansion
        # ====================================================

        context_start = time.perf_counter()

        context_documents = (
            self._expand_context(
                user_id=user_id,
                reranked_documents=reranked,
                max_chunks=context_max_chunks,
                window=context_window,
            )
        )

        context_latency = (
            time.perf_counter()
            - context_start
        )

        # ====================================================
        # 4. Build Prompt
        # ====================================================

        prompt_start = time.perf_counter()

        prompt = prompt_builder.build_prompt(
            query=query,
            retrieved_documents=context_documents,
        )

        prompt_latency = (
            time.perf_counter()
            - prompt_start
        )

        # ====================================================
        # 5. Generate Answer
        # ====================================================

        generation_start = time.perf_counter()

        answer = llm_service.generate(
            prompt
        )

        generation_latency = (
            time.perf_counter()
            - generation_start
        )

        # ====================================================
        # 6. Total Latency
        # ====================================================

        total_latency = (
            time.perf_counter()
            - total_start
        )

        # ====================================================
        # Retrieval Information
        # ====================================================

        weights = retrieval.get(
            "weights",
            {
                "bm25": 0.5,
                "dense": 0.5,
            },
        )

        retrieval_config = retrieval.get(
            "retrieval_config",
            {},
        )

        retrieval_metrics = retrieval.get(
            "retrieval_metrics",
            {},
        )

        model_name = retrieval.get(
            "model"
        )

        # ====================================================
        # Response
        # ====================================================

        return {

            "query":
                query,

            "answer":
                answer,

            # ------------------------------------------------
            # Adaptive Retrieval
            # ------------------------------------------------

            "weights":
                weights,

            "model":
                model_name,

            "retrieval_config":
                retrieval_config,

            "retrieval_metrics":
                retrieval_metrics,

            # ------------------------------------------------
            # Models
            # ------------------------------------------------

            "models": {

                "embedding":
                    "all-MiniLM-L6-v2",

                "adaptive_predictor":
                    model_name,

                "reranker":
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",

                "generation":
                    getattr(
                        llm_service,
                        "model",
                        "qwen2.5:7b",
                    ),
            },

            # ------------------------------------------------
            # Reranking
            # ------------------------------------------------

            "reranking": {

                "applied":
                    True,

                "method":
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",

                "candidate_count":
                    len(candidates),

                "reranked_count":
                    len(reranked),

                "latency_ms":
                    round(
                        reranking_latency * 1000,
                        2,
                    ),
            },

            # ------------------------------------------------
            # Context Selection
            # ------------------------------------------------

            "context": {

                "initial_reranked_chunks":
                    len(reranked),

                "selected_chunks":
                    len(context_documents),

                "selection_method":
                    "adjacent_chunk_expansion",

                "window":
                    context_window,

                "selection_latency_ms":
                    round(
                        context_latency * 1000,
                        2,
                    ),
            },

            # ------------------------------------------------
            # Documents
            # ------------------------------------------------

            "documents":
                context_documents,

            # ------------------------------------------------
            # Latency
            # ------------------------------------------------

            "latency": {

                "retrieval_ms":
                    round(
                        retrieval_latency * 1000,
                        2,
                    ),

                "reranking_ms":
                    round(
                        reranking_latency * 1000,
                        2,
                    ),

                "context_selection_ms":
                    round(
                        context_latency * 1000,
                        2,
                    ),

                "prompt_ms":
                    round(
                        prompt_latency * 1000,
                        2,
                    ),

                "generation_ms":
                    round(
                        generation_latency * 1000,
                        2,
                    ),

                "total_ms":
                    round(
                        total_latency * 1000,
                        2,
                    ),
            },
        }


# ============================================================
# Singleton
# ============================================================

user_rag_pipeline = UserRAGPipeline()


# ============================================================
# CLI Test
# ============================================================

if __name__ == "__main__":

    while True:

        query = input(
            "\nQuery : "
        )

        if query.lower() == "exit":
            break

        response = (
            user_rag_pipeline.answer(
                user_id="demo_user",
                query=query,
                retrieval_candidate_k=50,
                rerank_top_k=5,
                context_max_chunks=8,
                context_window=2,
            )
        )

        print()
        print("=" * 100)
        print("FINAL ANSWER")
        print("=" * 100)

        print(
            response["answer"]
        )

        print()
        print("=" * 100)
        print("ADAPTIVE WEIGHTS")
        print("=" * 100)

        print(
            response["weights"]
        )

        print()
        print("=" * 100)
        print("CONTEXT")
        print("=" * 100)

        print(
            response["context"]
        )

        print()
        print("=" * 100)
        print("LATENCY")
        print("=" * 100)

        for key, value in response[
            "latency"
        ].items():

            print(
                f"{key:<25}: {value} ms"
            )
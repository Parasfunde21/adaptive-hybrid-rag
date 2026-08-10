from services.logging_service import (
    logging_service
)


logging_service.log_retrieval(

    query="test query",

    bm25_weight=0.3,

    dense_weight=0.7,

    retrieval_features={

        "bm25_top_score": 5.2,

        "bm25_mean_score": 3.1,

        "bm25_score_gap": 0.4,

        "dense_top_similarity": 0.72,

        "dense_mean_similarity": 0.61,

        "dense_similarity_gap": 0.08,

        "retrieval_overlap_ratio": 0.4,

        "retrieval_rank_agreement": 0.8,

        "bm25_result_count": 10,

        "dense_result_count": 10

    },

    model_name="Extra Trees",

    retrieval_latency=0.012,

    fusion_latency=0.002,

    total_latency=0.014,

    result_count=10
)

print(
    "Logging test completed."
)
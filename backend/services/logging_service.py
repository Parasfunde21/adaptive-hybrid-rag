import json
import os
from datetime import datetime, timezone


class LoggingService:

    def __init__(self):

        self.log_directory = "logs"

        self.log_file = os.path.join(
            self.log_directory,
            "retrieval_logs.jsonl"
        )

        os.makedirs(
            self.log_directory,
            exist_ok=True
        )

    # =====================================================
    # Log Retrieval Request
    # =====================================================

    def log_retrieval(
        self,
        query,
        bm25_weight,
        dense_weight,
        retrieval_features,
        model_name,
        retrieval_latency,
        fusion_latency,
        total_latency,
        result_count
    ):

        log_entry = {

            "timestamp":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "query":
                query,

            "model":
                model_name,

            "weights": {

                "bm25":
                    round(
                        float(bm25_weight),
                        6
                    ),

                "dense":
                    round(
                        float(dense_weight),
                        6
                    )
            },

            "retrieval": {

                "bm25_top_score":
                    retrieval_features.get(
                        "bm25_top_score",
                        0.0
                    ),

                "bm25_mean_score":
                    retrieval_features.get(
                        "bm25_mean_score",
                        0.0
                    ),

                "bm25_score_gap":
                    retrieval_features.get(
                        "bm25_score_gap",
                        0.0
                    ),

                "dense_top_similarity":
                    retrieval_features.get(
                        "dense_top_similarity",
                        0.0
                    ),

                "dense_mean_similarity":
                    retrieval_features.get(
                        "dense_mean_similarity",
                        0.0
                    ),

                "dense_similarity_gap":
                    retrieval_features.get(
                        "dense_similarity_gap",
                        0.0
                    ),

                "overlap_ratio":
                    retrieval_features.get(
                        "retrieval_overlap_ratio",
                        0.0
                    ),

                "rank_agreement":
                    retrieval_features.get(
                        "retrieval_rank_agreement",
                        0.0
                    ),

                "bm25_result_count":
                    retrieval_features.get(
                        "bm25_result_count",
                        0
                    ),

                "dense_result_count":
                    retrieval_features.get(
                        "dense_result_count",
                        0
                    )
            },

            "performance": {

                "retrieval_latency_ms":
                    round(
                        retrieval_latency * 1000,
                        3
                    ),

                "fusion_latency_ms":
                    round(
                        fusion_latency * 1000,
                        3
                    ),

                "total_latency_ms":
                    round(
                        total_latency * 1000,
                        3
                    )
            },

            "result_count":
                result_count
        }

        with open(
            self.log_file,
            "a",
            encoding="utf-8"
        ) as file:

            file.write(
                json.dumps(
                    log_entry,
                    ensure_ascii=False
                )
                + "\n"
            )


logging_service = LoggingService()
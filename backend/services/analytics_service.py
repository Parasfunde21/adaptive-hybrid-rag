import json
import os
import statistics


class AnalyticsService:

    def __init__(self):

        self.log_file = os.path.join(
            "logs",
            "retrieval_logs.jsonl"
        )

    # =====================================================
    # Load Logs
    # =====================================================

    def load_logs(self):

        if not os.path.exists(
            self.log_file
        ):
            return []

        logs = []

        with open(
            self.log_file,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                try:

                    logs.append(
                        json.loads(line)
                    )

                except json.JSONDecodeError:

                    continue

        return logs

    # =====================================================
    # Basic Statistics
    # =====================================================

    def get_summary(self):

        logs = self.load_logs()

        if not logs:

            return {

                "total_queries": 0,

                "average_latency_ms": 0,

                "average_retrieval_latency_ms": 0,

                "average_fusion_latency_ms": 0,

                "average_bm25_weight": 0,

                "average_dense_weight": 0,

                "average_overlap_ratio": 0,

                "average_rank_agreement": 0,

                "average_result_count": 0

            }

        total_queries = len(logs)

        total_latency = [

            log[
                "performance"
            ][
                "total_latency_ms"
            ]

            for log in logs

            if "performance" in log
        ]

        retrieval_latency = [

            log[
                "performance"
            ][
                "retrieval_latency_ms"
            ]

            for log in logs

            if "performance" in log
        ]

        fusion_latency = [

            log[
                "performance"
            ][
                "fusion_latency_ms"
            ]

            for log in logs

            if "performance" in log
        ]

        bm25_weights = [

            log[
                "weights"
            ][
                "bm25"
            ]

            for log in logs

            if "weights" in log
        ]

        dense_weights = [

            log[
                "weights"
            ][
                "dense"
            ]

            for log in logs

            if "weights" in log
        ]

        overlap_ratios = [

            log[
                "retrieval"
            ][
                "overlap_ratio"
            ]

            for log in logs

            if "retrieval" in log
        ]

        rank_agreements = [

            log[
                "retrieval"
            ][
                "rank_agreement"
            ]

            for log in logs

            if "retrieval" in log
        ]

        result_counts = [

            log[
                "result_count"
            ]

            for log in logs

            if "result_count" in log
        ]

        return {

            "total_queries":
                total_queries,

            "average_latency_ms":
                round(
                    statistics.mean(
                        total_latency
                    ),
                    3
                ),

            "average_retrieval_latency_ms":
                round(
                    statistics.mean(
                        retrieval_latency
                    ),
                    3
                ),

            "average_fusion_latency_ms":
                round(
                    statistics.mean(
                        fusion_latency
                    ),
                    3
                ),

            "average_bm25_weight":
                round(
                    statistics.mean(
                        bm25_weights
                    ),
                    4
                ),

            "average_dense_weight":
                round(
                    statistics.mean(
                        dense_weights
                    ),
                    4
                ),

            "average_overlap_ratio":
                round(
                    statistics.mean(
                        overlap_ratios
                    ),
                    4
                ),

            "average_rank_agreement":
                round(
                    statistics.mean(
                        rank_agreements
                    ),
                    4
                ),

            "average_result_count":
                round(
                    statistics.mean(
                        result_counts
                    ),
                    2
                )

        }

    # =====================================================
    # Weight Distribution
    # =====================================================

    def get_weight_distribution(self):

        logs = self.load_logs()

        if not logs:
            return {}

        distribution = {

            "bm25_dominant": 0,

            "dense_dominant": 0,

            "balanced": 0

        }

        for log in logs:

            bm25 = log[
                "weights"
            ][
                "bm25"
            ]

            dense = log[
                "weights"
            ][
                "dense"
            ]

            if bm25 >= 0.6:

                distribution[
                    "bm25_dominant"
                ] += 1

            elif dense >= 0.6:

                distribution[
                    "dense_dominant"
                ] += 1

            else:

                distribution[
                    "balanced"
                ] += 1

        return distribution

    # =====================================================
    # Model Usage
    # =====================================================

    def get_model_usage(self):

        logs = self.load_logs()

        usage = {}

        for log in logs:

            model = log.get(
                "model",
                "unknown"
            )

            usage[model] = (
                usage.get(
                    model,
                    0
                )
                + 1
            )

        return usage

    # =====================================================
    # Recent Queries
    # =====================================================

    def get_recent_queries(
        self,
        limit=10
    ):

        logs = self.load_logs()

        recent = logs[
            -limit:
        ]

        return [

            {

                "timestamp":
                    log.get(
                        "timestamp"
                    ),

                "query":
                    log.get(
                        "query"
                    ),

                "bm25_weight":
                    log[
                        "weights"
                    ][
                        "bm25"
                    ],

                "dense_weight":
                    log[
                        "weights"
                    ][
                        "dense"
                    ],

                "latency_ms":
                    log[
                        "performance"
                    ][
                        "total_latency_ms"
                    ]

            }

            for log in reversed(
                recent
            )

        ]


analytics_service = (
    AnalyticsService()
)


if __name__ == "__main__":

    print()
    print("=" * 70)
    print(
        "RAG ANALYTICS"
    )
    print("=" * 70)

    print()

    print(
        "SUMMARY"
    )

    print(
        analytics_service
        .get_summary()
    )

    print()

    print(
        "WEIGHT DISTRIBUTION"
    )

    print(
        analytics_service
        .get_weight_distribution()
    )

    print()

    print(
        "MODEL USAGE"
    )

    print(
        analytics_service
        .get_model_usage()
    )

    print()

    print(
        "RECENT QUERIES"
    )

    for query in (
        analytics_service
        .get_recent_queries()
    ):

        print(
            query
        )
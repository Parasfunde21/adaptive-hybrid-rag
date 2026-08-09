from evaluation.evaluator import evaluator


def print_scores(name, scores):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(f"Precision@10 : {scores['precision']:.4f}")
    print(f"Recall@10    : {scores['recall']:.4f}")
    print(f"MRR          : {scores['mrr']:.4f}")
    print(f"nDCG@10      : {scores['ndcg']:.4f}")
    print(f"Latency      : {scores['latency']:.4f} sec")


def compare_results(results):

    print()
    print("=" * 90)
    print("OVERALL COMPARISON")
    print("=" * 90)

    print(
        f"{'Method':<30}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'MRR':>12}"
        f"{'nDCG':>12}"
        f"{'Latency':>15}"
    )

    print("-" * 90)

    for name, score in results.items():

        print(
            f"{name:<30}"
            f"{score['precision']:>12.4f}"
            f"{score['recall']:>12.4f}"
            f"{score['mrr']:>12.4f}"
            f"{score['ndcg']:>12.4f}"
            f"{score['latency']:>15.4f}"
        )


if __name__ == "__main__":

    bm25 = evaluator.evaluate_bm25()

    dense = evaluator.evaluate_dense()

    rrf = evaluator.evaluate_rrf()

    adaptive = evaluator.evaluate_adaptive()

    print_scores(
        "BM25",
        bm25
    )

    print_scores(
        "Dense Retrieval",
        dense
    )

    print_scores(
        "Reciprocal Rank Fusion (RRF)",
        rrf
    )

    print_scores(
        "Adaptive Hybrid Retrieval",
        adaptive
    )

    compare_results({

        "BM25": bm25,

        "Dense": dense,

        "RRF": rrf,

        "Adaptive": adaptive

    })
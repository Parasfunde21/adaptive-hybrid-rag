import os
import joblib
import numpy as np
import pandas as pd


MODEL_PATH = os.path.join(
    "models",
    "adaptive_weight_model_v4_rich.pkl"
)


class AdaptivePredictorV4:

    def __init__(self):

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"V4 model not found: {MODEL_PATH}"
            )

        with open(
            MODEL_PATH,
            "rb"
        ) as file:

            self.model = joblib.load(file)

        self.features = [
            "bm25_top_score",
            "bm25_mean_score",
            "bm25_score_std",
            "bm25_score_gap",
            "bm25_top_mean_ratio",

            "dense_top_similarity",
            "dense_mean_similarity",
            "dense_similarity_std",
            "dense_similarity_gap",
            "dense_top_mean_ratio",

            "overlap_ratio",
            "rank_agreement",

            "query_token_count",
            "query_unique_token_count",
            "query_avg_token_length",

            "bm25_top_lexical_coverage",
            "dense_top_lexical_coverage",
        ]

    def predict(
        self,
        query,
        retrieval_features
    ):

        row = [
            retrieval_features.get(
                feature,
                np.nan
            )
            for feature in self.features
        ]

        x = pd.DataFrame(
            [row],
            columns=self.features
        )

        predicted_bm25_weight = float(
            self.model.predict(x)[0]
        )

        predicted_bm25_weight = float(
            np.clip(
                predicted_bm25_weight,
                0.0,
                1.0
            )
        )

        predicted_dense_weight = (
            1.0 -
            predicted_bm25_weight
        )

        return {
            "bm25_weight":
                round(
                    predicted_bm25_weight,
                    6
                ),

            "dense_weight":
                round(
                    predicted_dense_weight,
                    6
                ),

            "model":
                "Extra Trees V4 Rich"
        }


adaptive_predictor_v4 = (
    AdaptivePredictorV4()
)


if __name__ == "__main__":

    print(
        "V4 Rich predictor loaded successfully."
    )
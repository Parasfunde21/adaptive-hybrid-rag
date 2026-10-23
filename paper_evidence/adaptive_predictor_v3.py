import os
import pickle
import numpy as np
import pandas as pd

from services.feature_extractor import feature_extractor


MODEL_PATH = os.path.join(
    "models",
    "adaptive_weight_model_v3.pkl"
)


class AdaptivePredictorV3:

    def __init__(self):

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"V3 model not found: {MODEL_PATH}"
            )

        with open(
            MODEL_PATH,
            "rb"
        ) as file:

            package = pickle.load(file)

        self.model = package["model"]
        self.imputer = package["imputer"]
        self.features = package["features"]
        self.model_name = package["model_name"]

    # =====================================================
    # Query Features
    # =====================================================

    def extract_query_features(self, query):

        features = feature_extractor.extract(
            query
        )

        return {
            f"q_{key}": value
            for key, value in features.items()
        }

    # =====================================================
    # Predict From Existing Retrieval Results
    # =====================================================

    def predict(
        self,
        query,
        retrieval_features
    ):

        query_features = (
            self.extract_query_features(
                query
            )
        )

        combined = {}

        combined.update(
            query_features
        )

        combined.update(
            {
                f"r_{key}": value
                for key, value
                in retrieval_features.items()
            }
        )

        row = [
            combined.get(
                feature,
                np.nan
            )
            for feature in self.features
        ]

        x = pd.DataFrame(
            [row],
            columns=self.features
        )

        x = self.imputer.transform(
            x
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
            1.0
            -
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
                self.model_name
        }


adaptive_predictor_v3 = (
    AdaptivePredictorV3()
)


if __name__ == "__main__":

    print(
        "V3 predictor loaded successfully."
    )
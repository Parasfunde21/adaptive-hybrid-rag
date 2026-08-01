import os
import joblib
import numpy as np

from services.feature_extractor import feature_extractor


MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "adaptive_weight_model.pkl"
)


class AdaptiveWeightPredictor:

    def __init__(self):

        self.model = joblib.load(MODEL_PATH)

    def predict(self, query: str):

        features = feature_extractor.extract(query)

        feature_vector = np.array([
            list(features.values())
        ])

        bm25_weight = float(
            self.model.predict(feature_vector)[0]
        )

        # Keep prediction within valid probability range
        bm25_weight = max(
            0.0,
            min(1.0, bm25_weight)
        )

        dense_weight = 1.0 - bm25_weight

        return {

            "query": query,

            "bm25_weight": round(
                bm25_weight,
                4
            ),

            "dense_weight": round(
                dense_weight,
                4
            ),

            "features": features

        }


adaptive_predictor = AdaptiveWeightPredictor()


if __name__ == "__main__":

    while True:

        query = input("\nQuery : ")

        if query.lower() == "exit":
            break

        prediction = adaptive_predictor.predict(query)

        print("\nAdaptive Weights")
        print("-" * 40)

        print(
            f"BM25 : {prediction['bm25_weight']}"
        )

        print(
            f"Dense: {prediction['dense_weight']}"
        )

        print("\nExtracted Features")

        for key, value in prediction["features"].items():

            print(f"{key:28}: {value}")
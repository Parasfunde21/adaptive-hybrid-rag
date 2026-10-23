import os
import joblib
import numpy as np

from services.retrieval_feature_service import V4_RICH_FEATURES

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

        self.features = list(V4_RICH_FEATURES)

    def predict(
        self,
        query,
        retrieval_features
    ):

        missing = [
            feature
            for feature in self.features
            if feature not in retrieval_features
        ]

        if missing:
            raise ValueError(
                "Missing required V4 retrieval features: "
                f"{', '.join(missing)}"
            )

        row = []
        for feature in self.features:
            value = retrieval_features[feature]
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"V4 retrieval feature '{feature}' must be numeric."
                ) from error

            if not np.isfinite(numeric_value):
                raise ValueError(
                    f"V4 retrieval feature '{feature}' must be finite."
                )

            row.append(numeric_value)

        x = np.asarray([row], dtype=np.float64)

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
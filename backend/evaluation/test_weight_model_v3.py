import pickle

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


TEST_PATH = "training/ml_test_v2.csv"

MODEL_PATH = (
    "models/adaptive_weight_model_v3.pkl"
)

OUTPUT_PATH = (
    "evaluation/v3_test_predictions.csv"
)


def main():

    print()
    print("=" * 80)
    print("V3 ADAPTIVE WEIGHT MODEL — UNSEEN TEST EVALUATION")
    print("=" * 80)

    # -----------------------------------------------------
    # Load test dataset
    # -----------------------------------------------------

    test_df = pd.read_csv(
        TEST_PATH
    )

    print()
    print(
        f"Test samples : {len(test_df)}"
    )

    # -----------------------------------------------------
    # Load trained model
    # -----------------------------------------------------

    with open(
        MODEL_PATH,
        "rb"
    ) as file:

        package = pickle.load(
            file
        )

    model = package["model"]
    imputer = package["imputer"]
    feature_columns = package["features"]

    print(
        f"Model        : "
        f"{package['model_name']}"
    )

    # -----------------------------------------------------
    # Build test matrix
    # -----------------------------------------------------

    x_test = test_df[
        feature_columns
    ]

    y_test = test_df[
        "bm25_weight"
    ]

    # -----------------------------------------------------
    # Apply training imputer
    # -----------------------------------------------------

    x_test = imputer.transform(
        x_test
    )

    # -----------------------------------------------------
    # Predict
    # -----------------------------------------------------

    predictions = model.predict(
        x_test
    )

    predictions = np.clip(
        predictions,
        0.0,
        1.0
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("UNSEEN TEST RESULTS")
    print("=" * 80)

    print(
        f"RMSE : {rmse:.6f}"
    )

    print(
        f"MAE  : {mae:.6f}"
    )

    print(
        f"R²   : {r2:.6f}"
    )

    # -----------------------------------------------------
    # Prediction comparison
    # -----------------------------------------------------

    comparison = pd.DataFrame({

        "sample_index":
            range(len(test_df)),

        "actual_bm25_weight":
            y_test.values,

        "predicted_bm25_weight":
            predictions

    })

    comparison["absolute_error"] = (

        abs(

            comparison[
                "actual_bm25_weight"
            ]

            -

            comparison[
                "predicted_bm25_weight"
            ]

        )

    )

    # -----------------------------------------------------
    # Prediction examples
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("SAMPLE PREDICTIONS")
    print("=" * 80)

    print(
        comparison.head(20).to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # Prediction distribution
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("PREDICTION DISTRIBUTION")
    print("=" * 80)

    print(
        pd.Series(
            predictions,
            name="predicted_bm25_weight"
        ).describe()
    )

    # -----------------------------------------------------
    # Save predictions
    # -----------------------------------------------------

    comparison.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print(
        f"Saved predictions to: "
        f"{OUTPUT_PATH}"
    )

    print()


if __name__ == "__main__":

    main()
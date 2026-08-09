import os
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)

from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


warnings.filterwarnings("ignore")


# =========================================================
# Configuration
# =========================================================

TRAIN_PATH = "training/ml_train.csv"

VALIDATION_PATH = (
    "training/ml_validation.csv"
)

MODEL_DIRECTORY = "models"

TARGET_COLUMN = "bm25_weight"

RANDOM_STATE = 42


# =========================================================
# Feature Extraction
# =========================================================

def extract_features(query):

    from services.feature_extractor import feature_extractor

    # IMPORTANT:
    # Existing FeatureExtractor uses:
    #
    #     feature_extractor.extract(query)
    #
    # NOT extract_features(query)

    features = feature_extractor.extract(
        query
    )

    if not isinstance(features, dict):

        raise TypeError(
            "feature_extractor.extract() "
            "must return a dictionary."
        )

    return features


# =========================================================
# Build Feature Dataset
# =========================================================

def build_feature_dataset(dataframe):

    rows = []

    total = len(dataframe)

    for index, (_, sample) in enumerate(
        dataframe.iterrows(),
        start=1
    ):

        query = sample["query"]

        features = extract_features(
            query
        )

        row = {}

        # ---------------------------------------------
        # Convert all features to numeric values
        # ---------------------------------------------

        for key, value in features.items():

            try:

                row[key] = float(value)

            except (
                TypeError,
                ValueError
            ):

                row[key] = np.nan

        # ---------------------------------------------
        # Target
        # ---------------------------------------------

        row[TARGET_COLUMN] = float(
            sample[TARGET_COLUMN]
        )

        rows.append(row)

        # ---------------------------------------------
        # Progress
        # ---------------------------------------------

        if (
            index == 1
            or index % 100 == 0
            or index == total
        ):

            print(
                f"Extracted features "
                f"{index}/{total}"
            )

    return pd.DataFrame(rows)


# =========================================================
# Evaluate Model
# =========================================================

def evaluate_model(
    model,
    x_train,
    y_train,
    x_validation,
    y_validation
):

    # ---------------------------------------------
    # Train
    # ---------------------------------------------

    model.fit(
        x_train,
        y_train
    )

    # ---------------------------------------------
    # Predict
    # ---------------------------------------------

    predictions = model.predict(
        x_validation
    )

    # ---------------------------------------------
    # Weight must remain between 0 and 1
    # ---------------------------------------------

    predictions = np.clip(
        predictions,
        0.0,
        1.0
    )

    # ---------------------------------------------
    # Metrics
    # ---------------------------------------------

    rmse = np.sqrt(
        mean_squared_error(
            y_validation,
            predictions
        )
    )

    mae = mean_absolute_error(
        y_validation,
        predictions
    )

    r2 = r2_score(
        y_validation,
        predictions
    )

    return {

        "model": model,

        "rmse": rmse,

        "mae": mae,

        "r2": r2,

        "predictions": predictions

    }


# =========================================================
# Main
# =========================================================

def main():

    print()
    print("=" * 80)
    print(
        "PERFORMANCE-BASED ADAPTIVE "
        "WEIGHT MODEL COMPARISON"
    )
    print("=" * 80)

    # =====================================================
    # Load datasets
    # =====================================================

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    validation_df = pd.read_csv(
        VALIDATION_PATH
    )

    print()

    print(
        f"Training samples   : "
        f"{len(train_df)}"
    )

    print(
        f"Validation samples : "
        f"{len(validation_df)}"
    )

    # =====================================================
    # Extract training features
    # =====================================================

    print()
    print(
        "Extracting training features..."
    )
    print()

    train_features = build_feature_dataset(
        train_df
    )

    # =====================================================
    # Extract validation features
    # =====================================================

    print()
    print(
        "Extracting validation features..."
    )
    print()

    validation_features = build_feature_dataset(
        validation_df
    )

    # =====================================================
    # Feature columns
    # =====================================================

    feature_columns = [

        column

        for column in train_features.columns

        if column != TARGET_COLUMN

    ]

    x_train = train_features[
        feature_columns
    ]

    y_train = train_features[
        TARGET_COLUMN
    ]

    x_validation = validation_features[
        feature_columns
    ]

    y_validation = validation_features[
        TARGET_COLUMN
    ]

    # =====================================================
    # Feature information
    # =====================================================

    print()
    print(
        f"Number of features : "
        f"{len(feature_columns)}"
    )

    print()
    print(
        "Features:"
    )

    print("-" * 80)

    for feature in feature_columns:

        print(
            f"- {feature}"
        )

    # =====================================================
    # Handle missing values
    # =====================================================

    imputer = SimpleImputer(
        strategy="median"
    )

    x_train = imputer.fit_transform(
        x_train
    )

    x_validation = imputer.transform(
        x_validation
    )

    # =====================================================
    # Define candidate models
    # =====================================================

    models = {

        "Random Forest":

            RandomForestRegressor(

                n_estimators=300,

                max_depth=10,

                min_samples_leaf=3,

                random_state=RANDOM_STATE,

                n_jobs=-1

            ),

        "Extra Trees":

            ExtraTreesRegressor(

                n_estimators=300,

                max_depth=10,

                min_samples_leaf=3,

                random_state=RANDOM_STATE,

                n_jobs=-1

            ),

        "Gradient Boosting":

            GradientBoostingRegressor(

                n_estimators=200,

                learning_rate=0.05,

                max_depth=3,

                min_samples_leaf=5,

                random_state=RANDOM_STATE

            ),

        "Hist Gradient Boosting":

            HistGradientBoostingRegressor(

                max_iter=200,

                learning_rate=0.05,

                max_leaf_nodes=15,

                l2_regularization=0.1,

                random_state=RANDOM_STATE

            )

    }

    # =====================================================
    # Train models
    # =====================================================

    results = {}

    print()
    print("=" * 80)
    print(
        "MODEL COMPARISON"
    )
    print("=" * 80)

    for name, model in models.items():

        print()
        print(
            f"Training: {name}"
        )

        result = evaluate_model(

            model=model,

            x_train=x_train,

            y_train=y_train,

            x_validation=x_validation,

            y_validation=y_validation

        )

        results[name] = result

        print(
            f"RMSE : "
            f"{result['rmse']:.6f}"
        )

        print(
            f"MAE  : "
            f"{result['mae']:.6f}"
        )

        print(
            f"R²   : "
            f"{result['r2']:.6f}"
        )

    # =====================================================
    # Model Summary
    # =====================================================

    print()
    print("=" * 80)
    print(
        "MODEL SUMMARY"
    )
    print("=" * 80)

    print(
        f"{'Model':<28}"
        f"{'RMSE':>12}"
        f"{'MAE':>12}"
        f"{'R²':>12}"
    )

    print("-" * 64)

    for name, result in results.items():

        print(
            f"{name:<28}"
            f"{result['rmse']:>12.6f}"
            f"{result['mae']:>12.6f}"
            f"{result['r2']:>12.6f}"
        )

    # =====================================================
    # Select best model
    #
    # Primary:
    #     Lowest RMSE
    #
    # Secondary:
    #     Highest R²
    # =====================================================

    best_name = min(

        results,

        key=lambda name: (

            results[name]["rmse"],

            -results[name]["r2"]

        )

    )

    best_result = results[
        best_name
    ]

    best_model = best_result[
        "model"
    ]

    # =====================================================
    # Best model
    # =====================================================

    print()
    print("=" * 80)

    print(
        f"BEST MODEL: "
        f"{best_name}"
    )

    print("=" * 80)

    print(
        f"Validation RMSE : "
        f"{best_result['rmse']:.6f}"
    )

    print(
        f"Validation MAE  : "
        f"{best_result['mae']:.6f}"
    )

    print(
        f"Validation R²   : "
        f"{best_result['r2']:.6f}"
    )

    # =====================================================
    # Feature Importance
    # =====================================================

    if hasattr(
        best_model,
        "feature_importances_"
    ):

        importance = pd.DataFrame({

            "feature":
                feature_columns,

            "importance":
                best_model.feature_importances_

        })

        importance = importance.sort_values(

            "importance",

            ascending=False

        )

        print()
        print("=" * 80)
        print(
            "FEATURE IMPORTANCE"
        )
        print("=" * 80)

        for _, row in importance.iterrows():

            print(
                f"{row['feature']:<30}"
                f"{row['importance']:.6f}"
            )

    else:

        print()
        print(
            "Feature importance is not "
            "directly available for "
            f"{best_name}."
        )

    # =====================================================
    # Save model
    # =====================================================

    os.makedirs(
        MODEL_DIRECTORY,
        exist_ok=True
    )

    model_package = {

        "model":
            best_model,

        "imputer":
            imputer,

        "features":
            feature_columns,

        "target":
            TARGET_COLUMN,

        "model_name":
            best_name,

        "validation_rmse":
            best_result["rmse"],

        "validation_mae":
            best_result["mae"],

        "validation_r2":
            best_result["r2"]

    }

    output_path = os.path.join(

        MODEL_DIRECTORY,

        "adaptive_weight_model_v2.pkl"

    )

    with open(
        output_path,
        "wb"
    ) as file:

        pickle.dump(
            model_package,
            file
        )

    # =====================================================
    # Complete
    # =====================================================

    print()
    print("=" * 80)
    print(
        "MODEL SAVED"
    )
    print("=" * 80)

    print(
        f"Path: {output_path}"
    )

    print()


# =========================================================
# Entry Point
# =========================================================

if __name__ == "__main__":

    main()
"""
V4 Adaptive Weight Model Training

Target:
    oracle_bm25_weight

Features:
    bm25_score_gap
    dense_similarity_gap
    overlap_ratio
    rank_agreement

Important:
    The test set is kept untouched until final evaluation.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import train_test_split


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = (
    BASE_DIR
    / "training"
    / "oracle_weight_dataset.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR = (
    BASE_DIR
    / "evaluation"
    / "v4_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Features
# ============================================================

FEATURES = [
    "bm25_score_gap",
    "dense_similarity_gap",
    "overlap_ratio",
    "rank_agreement",
]

TARGET = "oracle_bm25_weight"


# ============================================================
# Utilities
# ============================================================

def regression_metrics(
    y_true,
    y_pred
):

    return {
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred
            )
        ),

        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred
                )
            )
        ),

        "r2": float(
            r2_score(
                y_true,
                y_pred
            )
        )
    }


def print_metrics(
    name,
    metrics
):

    print()
    print(name)

    print(
        f"  MAE  : {metrics['mae']:.4f}"
    )

    print(
        f"  RMSE : {metrics['rmse']:.4f}"
    )

    print(
        f"  R2   : {metrics['r2']:.4f}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 80)
    print("V4 ADAPTIVE WEIGHT MODEL TRAINING")
    print("=" * 80)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE
    )

    print()
    print(
        f"Dataset rows: {len(df)}"
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    required = FEATURES + [TARGET]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    df = df.dropna(
        subset=required
    ).reset_index(
        drop=True
    )

    X = df[
        FEATURES
    ].astype(float)

    y = df[
        TARGET
    ].astype(float)

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------
    #
    # First:
    #   70% train
    #   30% temporary
    #
    # Then temporary:
    #   50% validation
    #   50% test
    #
    # Final:
    #   70 / 15 / 15
    # --------------------------------------------------------

    indices = np.arange(
        len(df)
    )

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=42,
        shuffle=True
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=42,
        shuffle=True
    )

    X_train = X.iloc[
        train_idx
    ]

    y_train = y.iloc[
        train_idx
    ]

    X_val = X.iloc[
        val_idx
    ]

    y_val = y.iloc[
        val_idx
    ]

    X_test = X.iloc[
        test_idx
    ]

    y_test = y.iloc[
        test_idx
    ]

    print()
    print("DATA SPLIT")

    print(
        f"Train      : {len(X_train)}"
    )

    print(
        f"Validation : {len(X_val)}"
    )

    print(
        f"Test       : {len(X_test)}"
    )

    print()
    print("Oracle weight distribution:")

    print(
        "Train:"
    )

    print(
        y_train.value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        "Validation:"
    )

    print(
        y_val.value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        "Test:"
    )

    print(
        y_test.value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Model 1: Extra Trees
    # --------------------------------------------------------

    extra_trees = ExtraTreesRegressor(

        n_estimators=300,

        max_depth=None,

        min_samples_leaf=2,

        max_features=1.0,

        random_state=42,

        n_jobs=-1
    )

    extra_trees.fit(
        X_train,
        y_train
    )

    val_pred_et = np.clip(
        extra_trees.predict(X_val),
        0.0,
        1.0
    )

    et_val_metrics = regression_metrics(
        y_val,
        val_pred_et
    )

    # --------------------------------------------------------
    # Model 2: Random Forest
    # --------------------------------------------------------

    random_forest = RandomForestRegressor(

        n_estimators=300,

        max_depth=None,

        min_samples_leaf=2,

        max_features=1.0,

        random_state=42,

        n_jobs=-1
    )

    random_forest.fit(
        X_train,
        y_train
    )

    val_pred_rf = np.clip(
        random_forest.predict(X_val),
        0.0,
        1.0
    )

    rf_val_metrics = regression_metrics(
        y_val,
        val_pred_rf
    )

    # --------------------------------------------------------
    # Validation comparison
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)

    print_metrics(
        "Extra Trees",
        et_val_metrics
    )

    print_metrics(
        "Random Forest",
        rf_val_metrics
    )

    # --------------------------------------------------------
    # Select model
    # --------------------------------------------------------

    if (
        et_val_metrics["mae"]
        <=
        rf_val_metrics["mae"]
    ):

        selected_model = extra_trees

        selected_name = (
            "Extra Trees"
        )

        selected_val_pred = (
            val_pred_et
        )

        selected_val_metrics = (
            et_val_metrics
        )

    else:

        selected_model = random_forest

        selected_name = (
            "Random Forest"
        )

        selected_val_pred = (
            val_pred_rf
        )

        selected_val_metrics = (
            rf_val_metrics
        )

    print()
    print(
        f"Selected model: "
        f"{selected_name}"
    )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    test_pred = np.clip(
        selected_model.predict(
            X_test
        ),
        0.0,
        1.0
    )

    test_metrics = regression_metrics(
        y_test,
        test_pred
    )

    print()
    print("=" * 80)
    print("FINAL HELD-OUT TEST")
    print("=" * 80)

    print_metrics(
        selected_name,
        test_metrics
    )

    # --------------------------------------------------------
    # Compare predictions
    # --------------------------------------------------------

    comparison = df.iloc[
        test_idx
    ][
        ["query"]
    ].copy()

    comparison[
        "oracle_bm25_weight"
    ] = y_test.values

    comparison[
        "predicted_bm25_weight"
    ] = test_pred

    comparison[
        "absolute_error"
    ] = np.abs(
        comparison[
            "oracle_bm25_weight"
        ]
        -
        comparison[
            "predicted_bm25_weight"
        ]
    )

    comparison = comparison.sort_values(
        "absolute_error",
        ascending=False
    )

    print()
    print(
        "TEST PREDICTIONS"
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("FEATURE IMPORTANCE")
    print("=" * 80)

    importance = pd.DataFrame({

        "feature":
            FEATURES,

        "importance":
            selected_model.feature_importances_

    }).sort_values(
        "importance",
        ascending=False
    )

    print(
        importance.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR
        / "adaptive_weight_model_v4.pkl"
    )

    joblib.dump(
        selected_model,
        model_path
    )

    # --------------------------------------------------------
    # Save test predictions
    # --------------------------------------------------------

    prediction_path = (
        OUTPUT_DIR
        / "v4_test_predictions.csv"
    )

    comparison.to_csv(
        prediction_path,
        index=False
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {

        "model":
            selected_name,

        "features":
            FEATURES,

        "target":
            TARGET,

        "dataset_size":
            int(len(df)),

        "train_size":
            int(len(X_train)),

        "validation_size":
            int(len(X_val)),

        "test_size":
            int(len(X_test)),

        "validation_metrics":
            selected_val_metrics,

        "test_metrics":
            test_metrics,

        "random_state":
            42

    }

    metadata_path = (
        OUTPUT_DIR
        / "v4_model_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2
        )

    print()
    print("=" * 80)
    print("V4 MODEL SAVED")
    print("=" * 80)

    print()
    print(
        f"Model:\n{model_path}"
    )

    print()
    print(
        f"Predictions:\n{prediction_path}"
    )

    print()
    print(
        f"Metadata:\n{metadata_path}"
    )

    print()
    print("Training complete.")


if __name__ == "__main__":
    main()
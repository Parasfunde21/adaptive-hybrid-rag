from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = (
    BASE_DIR
    / "evaluation"
    / "v4_analysis"
    / "rich_features.csv"
)

MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "evaluation" / "v4_analysis"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


FEATURES = [
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

TARGET = "oracle_bm25_weight"


def metrics(y_true, y_pred):

    return {
        "mae": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(y_true, y_pred)
            )
        ),
        "r2": float(
            r2_score(y_true, y_pred)
        )
    }


def print_metrics(name, values):

    print()
    print(name)
    print(f"  MAE  : {values['mae']:.4f}")
    print(f"  RMSE : {values['rmse']:.4f}")
    print(f"  R2   : {values['r2']:.4f}")


def main():

    print()
    print("=" * 80)
    print("V4 RICH FEATURE MODEL")
    print("=" * 80)

    df = pd.read_csv(DATA_FILE)

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

    df = (
        df
        .dropna(subset=required)
        .reset_index(drop=True)
    )

    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(float)

    print()
    print(f"Rows: {len(df)}")
    print(f"Features: {len(FEATURES)}")

    # --------------------------------------------------------
    # 70 / 15 / 15 split
    # --------------------------------------------------------

    indices = np.arange(len(df))

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

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    print()
    print("SPLIT")
    print(f"Train      : {len(X_train)}")
    print(f"Validation : {len(X_val)}")
    print(f"Test       : {len(X_test)}")

    # --------------------------------------------------------
    # Extra Trees
    # --------------------------------------------------------

    extra_trees = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
    )

    extra_trees.fit(
        X_train,
        y_train
    )

    et_val_pred = np.clip(
        extra_trees.predict(X_val),
        0.0,
        1.0
    )

    et_val_metrics = metrics(
        y_val,
        et_val_pred
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    random_forest = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
    )

    random_forest.fit(
        X_train,
        y_train
    )

    rf_val_pred = np.clip(
        random_forest.predict(X_val),
        0.0,
        1.0
    )

    rf_val_metrics = metrics(
        y_val,
        rf_val_pred
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("VALIDATION")
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
    # Select model by validation MAE
    # --------------------------------------------------------

    if (
        et_val_metrics["mae"]
        <=
        rf_val_metrics["mae"]
    ):

        model = extra_trees
        model_name = "Extra Trees"

    else:

        model = random_forest
        model_name = "Random Forest"

    print()
    print(
        f"Selected model: {model_name}"
    )

    # --------------------------------------------------------
    # Held-out test
    # --------------------------------------------------------

    test_pred = np.clip(
        model.predict(X_test),
        0.0,
        1.0
    )

    test_metrics = metrics(
        y_test,
        test_pred
    )

    print()
    print("=" * 80)
    print("HELD-OUT TEST")
    print("=" * 80)

    print_metrics(
        model_name,
        test_metrics
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = df.iloc[test_idx][
        ["query"]
    ].copy()

    predictions[
        "oracle_bm25_weight"
    ] = y_test.values

    predictions[
        "predicted_bm25_weight"
    ] = test_pred

    predictions[
        "absolute_error"
    ] = np.abs(
        predictions[
            "oracle_bm25_weight"
        ]
        -
        predictions[
            "predicted_bm25_weight"
        ]
    )

    predictions = predictions.sort_values(
        "absolute_error",
        ascending=False
    )

    print()
    print("=" * 80)
    print("TEST PREDICTIONS")
    print("=" * 80)

    print(
        predictions.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_
    }).sort_values(
        "importance",
        ascending=False
    )

    print()
    print("=" * 80)
    print("FEATURE IMPORTANCE")
    print("=" * 80)

    print(
        importance.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR
        / "adaptive_weight_model_v4_rich.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    prediction_path = (
        OUTPUT_DIR
        / "v4_rich_test_predictions.csv"
    )

    predictions.to_csv(
        prediction_path,
        index=False
    )

    importance_path = (
        OUTPUT_DIR
        / "v4_rich_feature_importance.csv"
    )

    importance.to_csv(
        importance_path,
        index=False
    )

    metadata = {
        "model": model_name,
        "features": FEATURES,
        "target": TARGET,
        "rows": int(len(df)),
        "train_size": int(len(X_train)),
        "validation_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "validation_metrics":
            (
                et_val_metrics
                if model_name == "Extra Trees"
                else rf_val_metrics
            ),
        "test_metrics": test_metrics,
        "random_state": 42
    }

    metadata_path = (
        OUTPUT_DIR
        / "v4_rich_model_metadata.json"
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
    print("MODEL SAVED")
    print("=" * 80)

    print(
        f"\nModel:\n{model_path}"
    )

    print(
        f"\nPredictions:\n{prediction_path}"
    )

    print(
        f"\nFeature importance:\n{importance_path}"
    )

    print(
        f"\nMetadata:\n{metadata_path}"
    )

    print()
    print("Training complete.")


if __name__ == "__main__":
    main()
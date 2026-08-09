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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

TRAIN_PATH = "training/ml_train_v2.csv"
VALIDATION_PATH = "training/ml_validation_v2.csv"
MODEL_DIRECTORY = "models"
TARGET_COLUMN = "bm25_weight"
RANDOM_STATE = 42


def build_feature_dataset(dataframe):
    rows = []
    total = len(dataframe)

    for index, (_, sample) in enumerate(dataframe.iterrows(), start=1):
        row = {}

        for column, value in sample.items():
            if column == TARGET_COLUMN:
                continue

            try:
                row[column] = float(value)
            except (TypeError, ValueError):
                row[column] = np.nan

        row[TARGET_COLUMN] = float(sample[TARGET_COLUMN])
        rows.append(row)

        if index == 1 or index % 100 == 0 or index == total:
            print(f"Loaded features {index}/{total}")

    return pd.DataFrame(rows)


def evaluate_model(model, x_train, y_train, x_validation, y_validation):
    model.fit(x_train, y_train)
    predictions = np.clip(model.predict(x_validation), 0.0, 1.0)

    rmse = np.sqrt(mean_squared_error(y_validation, predictions))
    mae = mean_absolute_error(y_validation, predictions)
    r2 = r2_score(y_validation, predictions)

    return {
        "model": model,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "predictions": predictions,
    }


def main():
    print()
    print("=" * 80)
    print("V2 PERFORMANCE-BASED ADAPTIVE WEIGHT MODEL COMPARISON")
    print("=" * 80)

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)

    print(f"\nTraining samples   : {len(train_df)}")
    print(f"Validation samples : {len(validation_df)}")

    print("\nBuilding training feature matrix...")
    train_features = build_feature_dataset(train_df)

    print("\nBuilding validation feature matrix...")
    validation_features = build_feature_dataset(validation_df)

    feature_columns = [c for c in train_features.columns if c != TARGET_COLUMN]

    x_train = train_features[feature_columns]
    y_train = train_features[TARGET_COLUMN]

    x_validation = validation_features[feature_columns]
    y_validation = validation_features[TARGET_COLUMN]

    print(f"\nNumber of features : {len(feature_columns)}")
    print("\nFeatures:")
    print("-" * 80)
    for feature in feature_columns:
        print(f"- {feature}")

    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(x_train)
    x_validation = imputer.transform(x_validation)

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=3,
            random_state=RANDOM_STATE
        ),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.03,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=RANDOM_STATE
        ),
    }

    results = {}

    print()
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    for name, model in models.items():
        print(f"\nTraining: {name}")
        result = evaluate_model(model, x_train, y_train, x_validation, y_validation)
        results[name] = result

        print(f"RMSE : {result['rmse']:.6f}")
        print(f"MAE  : {result['mae']:.6f}")
        print(f"R²   : {result['r2']:.6f}")

    print()
    print("=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    print(f"{'Model':<28}{'RMSE':>12}{'MAE':>12}{'R²':>12}")
    print("-" * 64)

    for name, result in results.items():
        print(
            f"{name:<28}"
            f"{result['rmse']:>12.6f}"
            f"{result['mae']:>12.6f}"
            f"{result['r2']:>12.6f}"
        )

    best_name = min(
        results,
        key=lambda name: (results[name]["rmse"], -results[name]["r2"])
    )
    best_result = results[best_name]
    best_model = best_result["model"]

    print()
    print("=" * 80)
    print(f"BEST MODEL: {best_name}")
    print("=" * 80)
    print(f"Validation RMSE : {best_result['rmse']:.6f}")
    print(f"Validation MAE  : {best_result['mae']:.6f}")
    print(f"Validation R²   : {best_result['r2']:.6f}")

    if hasattr(best_model, "feature_importances_"):
        importance = pd.DataFrame({
            "feature": feature_columns,
            "importance": best_model.feature_importances_
        }).sort_values("importance", ascending=False)

        print()
        print("=" * 80)
        print("FEATURE IMPORTANCE")
        print("=" * 80)
        for _, row in importance.iterrows():
            print(f"{row['feature']:<35}{row['importance']:.6f}")

    os.makedirs(MODEL_DIRECTORY, exist_ok=True)

    model_package = {
        "model": best_model,
        "imputer": imputer,
        "features": feature_columns,
        "target": TARGET_COLUMN,
        "model_name": best_name,
        "validation_rmse": best_result["rmse"],
        "validation_mae": best_result["mae"],
        "validation_r2": best_result["r2"],
    }

    output_path = os.path.join(MODEL_DIRECTORY, "adaptive_weight_model_v3.pkl")

    with open(output_path, "wb") as f:
        pickle.dump(model_package, f)

    print()
    print("=" * 80)
    print("MODEL SAVED")
    print("=" * 80)
    print(f"Path: {output_path}")
    print()


if __name__ == "__main__":
    main()
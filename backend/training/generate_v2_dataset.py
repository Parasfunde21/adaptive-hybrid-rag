import pandas as pd

from training.retrieval_features import retrieval_feature_extractor
from services.feature_extractor import feature_extractor


INPUT_TRAIN_PATH = "training/ml_train.csv"
INPUT_VALIDATION_PATH = "training/ml_validation.csv"
INPUT_TEST_PATH = "training/ml_test.csv"

OUTPUT_TRAIN_PATH = "training/ml_train_v2.csv"
OUTPUT_VALIDATION_PATH = "training/ml_validation_v2.csv"
OUTPUT_TEST_PATH = "training/ml_test_v2.csv"

TARGET_COLUMN = "bm25_weight"


def combine_features(query):
    query_features = feature_extractor.extract(query)
    retrieval_features = retrieval_feature_extractor.extract(query)

    combined = {}

    for key, value in query_features.items():
        combined[f"q_{key}"] = value

    for key, value in retrieval_features.items():
        combined[f"r_{key}"] = value

    return combined


def build_dataset(input_path, output_path):
    df = pd.read_csv(input_path)
    rows = []

    total = len(df)

    for index, sample in enumerate(df.itertuples(index=False), start=1):
        features = combine_features(sample.query)
        features[TARGET_COLUMN] = float(getattr(sample, TARGET_COLUMN))
        rows.append(features)

        if index == 1 or index % 50 == 0 or index == total:
            print(f"Processed {index}/{total}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)
    return out_df


if __name__ == "__main__":
    print()
    print("=" * 80)
    print("BUILDING V2 DATASETS")
    print("=" * 80)

    print("\nTraining set")
    train_df = build_dataset(INPUT_TRAIN_PATH, OUTPUT_TRAIN_PATH)

    print("\nValidation set")
    val_df = build_dataset(INPUT_VALIDATION_PATH, OUTPUT_VALIDATION_PATH)

    print("\nTest set")
    test_df = build_dataset(INPUT_TEST_PATH, OUTPUT_TEST_PATH)

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print(f"Train rows     : {len(train_df)}")
    print(f"Validation rows : {len(val_df)}")
    print(f"Test rows      : {len(test_df)}")
    print(f"Saved to       : {OUTPUT_TRAIN_PATH}, {OUTPUT_VALIDATION_PATH}, {OUTPUT_TEST_PATH}")
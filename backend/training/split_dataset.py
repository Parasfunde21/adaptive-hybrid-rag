import pandas as pd

from sklearn.model_selection import train_test_split


INPUT_PATH = "training/optimized_training_dataset.csv"

TRAIN_PATH = "training/ml_train.csv"
VALIDATION_PATH = "training/ml_validation.csv"
TEST_PATH = "training/ml_test.csv"


RANDOM_STATE = 42


def split_dataset():

    print()
    print("=" * 70)
    print("ML DATASET SPLIT")
    print("=" * 70)

    df = pd.read_csv(INPUT_PATH)

    print(
        f"Total samples : {len(df)}"
    )

    # -------------------------------------------------
    # Remove exact duplicate queries
    # -------------------------------------------------

    original_count = len(df)

    df = df.drop_duplicates(
        subset=["query"],
        keep="first"
    ).reset_index(drop=True)

    removed = original_count - len(df)

    print(
        f"Duplicate queries removed : {removed}"
    )

    print(
        f"Unique queries             : {len(df)}"
    )

    # -------------------------------------------------
    # First split:
    #
    # 70% train
    # 30% temporary
    # -------------------------------------------------

    train_df, temp_df = train_test_split(

        df,

        test_size=0.30,

        random_state=RANDOM_STATE

    )

    # -------------------------------------------------
    # Second split:
    #
    # 15% validation
    # 15% test
    # -------------------------------------------------

    validation_df, test_df = train_test_split(

        temp_df,

        test_size=0.50,

        random_state=RANDOM_STATE

    )

    # -------------------------------------------------
    # Save
    # -------------------------------------------------

    train_df.to_csv(
        TRAIN_PATH,
        index=False
    )

    validation_df.to_csv(
        VALIDATION_PATH,
        index=False
    )

    test_df.to_csv(
        TEST_PATH,
        index=False
    )

    # -------------------------------------------------
    # Report
    # -------------------------------------------------

    print()
    print("Dataset Split")
    print("-" * 70)

    print(
        f"Training   : {len(train_df)} "
        f"({len(train_df) / len(df) * 100:.2f}%)"
    )

    print(
        f"Validation : {len(validation_df)} "
        f"({len(validation_df) / len(df) * 100:.2f}%)"
    )

    print(
        f"Test       : {len(test_df)} "
        f"({len(test_df) / len(df) * 100:.2f}%)"
    )

    # -------------------------------------------------
    # Weight distributions
    # -------------------------------------------------

    print()
    print("Training BM25 Weight Distribution")
    print("-" * 70)

    print(
        train_df[
            "bm25_weight"
        ].value_counts().sort_index()
    )

    print()
    print("Validation BM25 Weight Distribution")
    print("-" * 70)

    print(
        validation_df[
            "bm25_weight"
        ].value_counts().sort_index()
    )

    print()
    print("Test BM25 Weight Distribution")
    print("-" * 70)

    print(
        test_df[
            "bm25_weight"
        ].value_counts().sort_index()
    )

    print()
    print("Saved:")
    print(TRAIN_PATH)
    print(VALIDATION_PATH)
    print(TEST_PATH)


if __name__ == "__main__":

    split_dataset()
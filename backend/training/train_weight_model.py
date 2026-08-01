import joblib
import pandas as pd

from pathlib import Path

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from config.settings import MODEL_DIR

# ---------------------------------------
# Load Dataset
# ---------------------------------------

dataset_path = Path(__file__).parent / "training_dataset.csv"

df = pd.read_csv(dataset_path)

# ---------------------------------------
# Features
# ---------------------------------------

X = df.drop(columns=[
    "bm25_weight",
    "dense_weight"
])

# Targets
y_bm25 = df["bm25_weight"]
y_dense = df["dense_weight"]

# ---------------------------------------
# Train Test Split
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_dense,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------
# Train Model
# ---------------------------------------

model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------------------
# Evaluation
# ---------------------------------------

predictions = model.predict(X_test)

print("\nModel Performance")

print("------------------------------")

print(
    "RMSE:",
    mean_squared_error(
        y_test,
        predictions
    ) ** 0.5
)

print(
    "R2:",
    r2_score(
        y_test,
        predictions
    )
)

# ---------------------------------------
# Save Model
# ---------------------------------------

MODEL_DIR.mkdir(
    exist_ok=True
)

model_path = MODEL_DIR / "adaptive_weight_model.pkl"

joblib.dump(
    model,
    model_path
)

print("\nSaved Model To:")

print(model_path)
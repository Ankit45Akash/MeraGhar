
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "Dataset" / "indian_house_prices.csv"
MODEL_PATH = BASE_DIR / "house_price_model.pkl"
COMPARISON_PATH = BASE_DIR / "model_comparison.json"

# --------------------------------------------------
# 2. EXPECTED DATASET COLUMNS
# --------------------------------------------------

FEATURE_COLUMNS = [
    "city",
    "locality",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "floors",
    "year_built",
    "parking",
    "facing",
    "lift",
    "furnishing",
    "garden",
]

TARGET_COLUMN = "price"

NUMERIC_FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "floors",
    "year_built",
    "parking",
]

CATEGORICAL_FEATURES = [
    "city",
    "locality",
    "facing",
    "lift",
    "furnishing",
    "garden",
]


def main():
    print("\n" + "=" * 55)
    print("          MERAGHAR - MODEL TRAINING")
    print("=" * 55)

    # --------------------------------------------------
    # 3. LOAD AND VALIDATE DATASET
    # --------------------------------------------------

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"\nDataset not found at:\n{DATASET_PATH}\n"
            "Check the filename and Dataset folder."
        )

    df = pd.read_csv(DATASET_PATH)
    df.columns = df.columns.str.strip()

    print(f"\nDataset loaded: {DATASET_PATH.name}")
    print(f"Original rows: {len(df)}")
    print(f"Columns found: {len(df.columns)}")

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "\nYour dataset is missing these required columns: "
            + ", ".join(missing_columns)
            + "\nPlease check the CSV column names."
        )

    # Keep only the columns required by this model.
    df = df[required_columns].copy()

    # Convert numerical fields into numbers.
    for column in NUMERIC_FEATURES + [TARGET_COLUMN]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Remove rows that have no usable target price.
    df = df.dropna(subset=[TARGET_COLUMN])

    # Prices must be positive.
    df = df[df[TARGET_COLUMN] > 0]

    # Treat blank categorical values as missing values.
    for column in CATEGORICAL_FEATURES:
        df[column] = df[column].replace(r"^\s*$", np.nan, regex=True)

    if len(df) < 20:
        raise ValueError(
            "Too few valid records for training. "
            "Please check your dataset."
        )

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    print(f"Usable rows: {len(df)}")
    print(f"Input features: {len(FEATURE_COLUMNS)}")
    print(f"Target: {TARGET_COLUMN}")

    # --------------------------------------------------
    # 4. SPLIT THE DATA
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print(f"\nTraining records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    # --------------------------------------------------
    # 5. PREPROCESSING
    # --------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    # --------------------------------------------------
    # 6. ALGORITHMS TO COMPARE
    # --------------------------------------------------

    algorithms = {
        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            max_depth=8,
            min_samples_leaf=4,
            random_state=42,
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=16,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=150,
            max_depth=16,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=2,
            random_state=42,
        ),
    }

    # --------------------------------------------------
    # 7. TRAIN AND EVALUATE EACH ALGORITHM
    # --------------------------------------------------

    results = []
    trained_models = {}

    print("\nTraining algorithms...\n")

    for name, algorithm in algorithms.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", algorithm),
            ]
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        results.append(
            {
                "algorithm": name,
                "r2_score": round(float(r2), 4),
                "mae": round(float(mae), 2),
                "rmse": round(float(rmse), 2),
            }
        )

        trained_models[name] = pipeline

        print(f"{name}")
        print(f"  R2 Score : {r2:.4f}")
        print(f"  MAE      : Rs. {mae:,.2f}")
        print(f"  RMSE     : Rs. {rmse:,.2f}")
        print("-" * 40)

    # --------------------------------------------------
    # 8. SELECT THE BEST MODEL
    # --------------------------------------------------

    # Highest R2 is preferred. MAE breaks ties.
    best_result = max(
        results,
        key=lambda item: (item["r2_score"], -item["mae"]),
    )

    best_name = best_result["algorithm"]
    best_model = trained_models[best_name]

    # --------------------------------------------------
    # 9. SAVE MODEL AND COMPARISON RESULTS
    # --------------------------------------------------

    model_package = {
        "model": best_model,
        "best_model": best_name,
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "metrics": best_result,
        "all_metrics": results,
        "currency": "INR",
        "sklearn_version": sklearn.__version__,
    }

    joblib.dump(model_package, MODEL_PATH)

    comparison_data = {
        "best_model": best_name,
        "target": TARGET_COLUMN,
        "currency": "INR",
        "dataset_rows": int(len(df)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "sklearn_version": sklearn.__version__,
        "results": results,
    }

    with open(COMPARISON_PATH, "w", encoding="utf-8") as file:
        json.dump(comparison_data, file, indent=4)

    # --------------------------------------------------
    # 10. FINAL SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 55)
    print("             TRAINING COMPLETED")
    print("=" * 55)

    print(f"\nBest model: {best_name}")
    print(f"R2 Score: {best_result['r2_score']:.4f}")
    print(f"MAE: Rs. {best_result['mae']:,.2f}")
    print(f"RMSE: Rs. {best_result['rmse']:,.2f}")

    print(f"\nModel saved to: {MODEL_PATH.name}")
    print(f"Comparison saved to: {COMPARISON_PATH.name}")

    print(
        "\nNote: These results evaluate the supplied dataset. "
        "If the data is synthetic, the scores do not establish "
        "accuracy on real property listings."
    )

    print("\nNext step: update app.py to use the saved model package.")
    print("=" * 55)


if __name__ == "__main__":
    main()

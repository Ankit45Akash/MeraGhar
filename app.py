
from pathlib import Path
import json

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "Dataset" / "indian_house_prices.csv"
MODEL_PATH = BASE_DIR / "house_price_model.pkl"
COMPARISON_PATH = BASE_DIR / "model_comparison.json"

FEATURE_COLUMNS = [
    "city", "locality", "bedrooms", "bathrooms", "sqft_living",
    "floors", "year_built", "parking", "facing", "lift",
    "furnishing", "garden"
]

CATEGORICAL_OPTIONS = {
    "city": ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Pune"],
    "facing": ["North", "South", "East", "West"],
    "lift": ["Yes", "No"],
    "furnishing": ["Furnished", "Semi-Furnished", "Unfurnished"],
    "garden": ["Yes", "No"],
}

NUMERIC_LIMITS = {
    "bedrooms": (1, 5),
    "bathrooms": (1, 4),
    "sqft_living": (300, 5000),
    "floors": (1, 3),
    "year_built": (1950, 2025),
    "parking": (0, 3),
}


def load_dataset():
    if not DATASET_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(DATASET_PATH)
    df.columns = df.columns.str.strip()
    return df


def load_comparison():
    if not COMPARISON_PATH.exists():
        return {"best_model": "Not available", "results": []}
    try:
        with open(COMPARISON_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {"best_model": "Not available", "results": []}


def load_model():
    if not MODEL_PATH.exists():
        return None, {}
    package = joblib.load(MODEL_PATH)
    if isinstance(package, dict) and "model" in package:
        return package["model"], package
    return package, {}


model, model_package = load_model()


@app.route("/")
def home():
    df = load_dataset()
    localities = (
        sorted(df["locality"].dropna().astype(str).unique().tolist())
        if not df.empty and "locality" in df.columns else []
    )
    return render_template(
        "index.html",
        cities=CATEGORICAL_OPTIONS["city"],
        localities=localities,
        options=CATEGORICAL_OPTIONS,
        numeric_limits=NUMERIC_LIMITS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            raise ValueError("Model is not available. Run python train_model.py first.")

        form = request.form
        input_data = {
            "city": form.get("city", "").strip(),
            "locality": form.get("locality", "").strip(),
            "bedrooms": int(form.get("bedrooms", "")),
            "bathrooms": int(form.get("bathrooms", "")),
            "sqft_living": float(form.get("sqft_living", "")),
            "floors": int(form.get("floors", "")),
            "year_built": int(form.get("year_built", "")),
            "parking": int(form.get("parking", "")),
            "facing": form.get("facing", "").strip(),
            "lift": form.get("lift", "").strip(),
            "furnishing": form.get("furnishing", "").strip(),
            "garden": form.get("garden", "").strip(),
        }

        for column, allowed in CATEGORICAL_OPTIONS.items():
            if input_data[column] not in allowed:
                raise ValueError(f"Please select a valid {column}.")

        df = load_dataset()
        if not df.empty and "locality" in df.columns:
            valid_localities = set(df["locality"].dropna().astype(str))
            if input_data["locality"] not in valid_localities:
                raise ValueError("Please select a locality from the dataset.")

        for column, (minimum, maximum) in NUMERIC_LIMITS.items():
            value = input_data[column]
            if not minimum <= value <= maximum:
                raise ValueError(
                    f"{column.replace('_', ' ').title()} must be between "
                    f"{minimum} and {maximum}."
                )

        prediction = float(model.predict(pd.DataFrame([input_data]))[0])
        if prediction <= 0:
            raise ValueError("The model returned an invalid estimate.")

        return render_template(
            "result.html",
            prediction=round(prediction),
            property_data=input_data,
            model_name=model_package.get("best_model", "Trained ML model"),
        )

    except (ValueError, TypeError, KeyError) as error:
        return render_template(
            "result.html",
            error=str(error),
            property_data=request.form,
        ), 400


@app.route("/dashboard")
def dashboard():
    df = load_dataset()
    if df.empty:
        return "Dataset not found.", 500

    prices = pd.to_numeric(df["price"], errors="coerce").dropna()
    comparison = load_comparison()

    stats = {
        "total_properties": len(df),
        "average_price": round(float(prices.mean())) if not prices.empty else 0,
        "median_price": round(float(prices.median())) if not prices.empty else 0,
        "cities": int(df["city"].nunique()),
        "best_model": comparison.get("best_model", "Not available"),
        "r2_score": model_package.get("metrics", {}).get("r2_score", 0),
    }

    city_counts = (
        df["city"].value_counts().rename_axis("city")
        .reset_index(name="count")
    )
    city_prices = (
        df.groupby("city", as_index=False)["price"].mean()
        .sort_values("price", ascending=False)
    )

    return render_template(
        "dashboard.html",
        stats=stats,
        comparison=comparison.get("results", []),
        city_counts=city_counts.to_dict(orient="records"),
        city_prices=city_prices.to_dict(orient="records"),
    )


@app.route("/dataset")
def dataset_explorer():
    df = load_dataset()
    if df.empty:
        return "Dataset not found.", 500

    return render_template(
        "dataset.html",
        columns=df.columns.tolist(),
        rows=df.head(100).to_dict(orient="records"),
        total_rows=len(df),
    )


@app.route("/algorithms")
def algorithms():
    comparison = load_comparison()
    return render_template(
        "algorithms.html",
        results=comparison.get("results", []),
        best_model=comparison.get("best_model", "Not available"),
    )


@app.route("/preprocessing")
def preprocessing():
    df = load_dataset()
    if df.empty:
        return "Dataset not found.", 500

    missing = df.isna().sum()
    missing_rows = [
        {"column": col, "missing": int(count)}
        for col, count in missing.items()
    ]

    duplicate_count = int(df.duplicated().sum())

    type_rows = [
        {"column": col, "dtype": str(dtype)}
        for col, dtype in df.dtypes.items()
    ]

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(exclude="number").columns.tolist()

    return render_template(
        "preprocessing.html",
        total_rows=len(df),
        total_columns=len(df.columns),
        duplicate_count=duplicate_count,
        missing_total=int(missing.sum()),
        missing_rows=missing_rows,
        type_rows=type_rows,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )


@app.route("/visualization")
def visualization():
    df = load_dataset()
    if df.empty:
        return "Dataset not found.", 500

    data = df.copy()
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data["sqft_living"] = pd.to_numeric(data["sqft_living"], errors="coerce")
    data["bedrooms"] = pd.to_numeric(data["bedrooms"], errors="coerce")
    data["year_built"] = pd.to_numeric(data["year_built"], errors="coerce")
    data = data.dropna(subset=["price"])

    # Average and median price by city.
    city_prices = (
        data.groupby("city")["price"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    # Average price by bedroom count.
    bedroom_prices = (
        data.dropna(subset=["bedrooms"])
        .groupby("bedrooms")["price"].mean()
        .reset_index()
        .sort_values("bedrooms")
    )

    # Price distribution in six data-based ranges.
    distribution = data["price"].value_counts(bins=6, sort=False).sort_index()
    distribution_labels = []
    distribution_counts = []
    for interval, count in distribution.items():
        distribution_labels.append(
            f"₹{interval.left / 100000:.0f}L–₹{interval.right / 100000:.0f}L"
        )
        distribution_counts.append(int(count))

    # Average price by furnishing category.
    furnishing_prices = (
        data.dropna(subset=["furnishing"])
        .groupby("furnishing")["price"].mean()
        .reset_index()
        .sort_values("price", ascending=False)
    )

    # Average price by living-area band.
    area_bins = [0, 1000, 2000, 3000, 4000, float("inf")]
    area_labels = ["≤1,000", "1,001–2,000", "2,001–3,000",
                   "3,001–4,000", "Above 4,000"]
    data["area_band"] = pd.cut(
        data["sqft_living"],
        bins=area_bins,
        labels=area_labels,
        include_lowest=True,
    )
    area_prices = (
        data.dropna(subset=["area_band"])
        .groupby("area_band", observed=False)["price"].mean()
        .reset_index()
    )

    # Average price by property age group.
    current_year = pd.Timestamp.now().year
    data["property_age"] = current_year - data["year_built"]
    data["age_band"] = pd.cut(
        data["property_age"],
        bins=[-1, 5, 15, 30, 1000],
        labels=["0–5 years", "6–15 years", "16–30 years", "Over 30 years"],
    )
    age_prices = (
        data.dropna(subset=["age_band"])
        .groupby("age_band", observed=False)["price"].mean()
        .reset_index()
    )

    return render_template(
        "visualization.html",
        total_rows=len(data),
        city_labels=city_prices["city"].astype(str).tolist(),
        city_means=[round(float(v), 2) for v in city_prices["mean"]],
        city_medians=[round(float(v), 2) for v in city_prices["median"]],
        city_counts=[int(v) for v in city_prices["count"]],
        bedroom_labels=[str(int(v)) for v in bedroom_prices["bedrooms"]],
        bedroom_means=[round(float(v), 2) for v in bedroom_prices["price"]],
        distribution_labels=distribution_labels,
        distribution_counts=distribution_counts,
        furnishing_labels=furnishing_prices["furnishing"].astype(str).tolist(),
        furnishing_means=[round(float(v), 2) for v in furnishing_prices["price"]],
        area_labels=area_prices["area_band"].astype(str).tolist(),
        area_means=[round(float(v), 2) for v in area_prices["price"]],
        age_labels=age_prices["age_band"].astype(str).tolist(),
        age_means=[round(float(v), 2) for v in age_prices["price"]],
    )


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)

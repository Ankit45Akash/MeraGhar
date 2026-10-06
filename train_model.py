import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("Dataset/indian_house_prices.csv")

print("Dataset loaded successfully!")
print("Total properties:", len(data))


# ==========================================
# 2. COLUMNS WE ARE USING
# ==========================================

features = [
    "Area",
    "Bedrooms",
    "City",
    "Parking",
    "Pool",
    "Resale"
]

target = "Price"

X = data[features]
y = data[target]


# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training properties:", len(X_train))
print("Testing properties:", len(X_test))


# ==========================================
# 4. DEFINE FEATURE TYPES
# ==========================================

numeric_features = [
    "Area",
    "Bedrooms",
    "Parking",
    "Pool",
    "Resale"
]

categorical_features = [
    "City"
]


# ==========================================
# 5. PREPROCESSING
# ==========================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ==========================================
# 6. RANDOM FOREST MODEL
# ==========================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        )
    ]
)


# ==========================================
# 7. TRAIN MODEL
# ==========================================

print("\nTraining MeraGhar model...")

model.fit(X_train, y_train)


# ==========================================
# 8. TEST MODEL
# ==========================================

predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)


# ==========================================
# 9. SHOW RESULTS
# ==========================================

print("\n================================")
print("       MERAGHAR MODEL")
print("================================")

print("R2 Score:", round(r2, 4))
print("MAE:", round(mae, 2))

print("================================")


# ==========================================
# 10. SAVE MODEL
# ==========================================

joblib.dump(
    {
        "model": model,
        "features": features
    },
    "house_price_model.pkl"
)

print("\nModel saved successfully!")
print("File: house_price_model.pkl")

print("\nFeatures used:")

for feature in features:
    print("-", feature)
import pandas as pd

# Load the dataset
data = pd.read_csv("Dataset/modified_data.csv")

print("\n===== DATASET SHAPE =====")
print(data.shape)

print("\n===== COLUMN NAMES =====")
print(data.columns.tolist())

print("\n===== FIRST 5 ROWS =====")
print(data.head())

print("\n===== DATA TYPES =====")
print(data.dtypes)

print("\n===== MISSING VALUES =====")
print(data.isnull().sum())

print("\n===== BASIC STATISTICS =====")
print(data.describe())


# ============================================================
# 1. IMPORTS
# ============================================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.neighbors import KNeighborsClassifier



# ============================================================
#  LOAD DATA
# ============================================================

df = pd.read_csv("C:/Users/User.DESKTOP-LUMR9IQ/Documents/14570063-ML-Assignment/credit-card-fraud-detection.csv")

print("Dataset shape:", df.shape)
print(df.head())

print("\nInitial Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================================
#   CLEANING THE DATA
# ============================================================

# Convert datetime column
df["transaction_datetime"] = pd.to_datetime(df["transaction_datetime"], errors="coerce")

# Convert cardholder_age and transaction_amount to numeric
df["cardholder_age"] = pd.to_numeric(df["cardholder_age"], errors="coerce")
df["transaction_amount"] = pd.to_numeric(df["transaction_amount"], errors="coerce")

# Fix the booleans

df['is_fraud'].info()
bool_cols = ["is_fraud", "is_international"]
for col in bool_cols:
    df[col] = df[col].astype(int)

# Fill missing transaction_state with "Intl"
df["transaction_state"] = df["transaction_state"].fillna("Intl")

# remove the missing transaction_city due to low count of missing values
df = df.dropna(subset=["transaction_city"])


# Remove Identifier Columns
id_cols = [ "transaction_id", "card_number", "cardholder_id", "merchant_id"]
df = df.drop(columns=id_cols)

# Remove Duplicates
df = df.drop_duplicates()


# ============================================================
#  FEATURE ENGINEERING
# ============================================================

df["year"] = df["transaction_datetime"].dt.year
df["month"] = df["transaction_datetime"].dt.month
df["day"] = df["transaction_datetime"].dt.day
df["hour"] = df["transaction_datetime"].dt.hour
df["dayofweek"] = df["transaction_datetime"].dt.dayofweek

# ============================================================
#  SPLIT FEATURES / TARGET
# ============================================================

y = df["is_fraud"]
X = df.drop(columns=["is_fraud"])

# ============================================================
# DETECT COLUMN TYPES
# ============================================================

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)


# ============================================================
# PREPROCESSING PIPELINE
# ============================================================

numeric_pipeline = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

# ============================================================
# ML MODEL + SMOTE OVERSAMPLING PIPELINE
# ============================================================

model = KNeighborsClassifier()

pipeline = ImbPipeline(steps=[
    ("preprocess", preprocessor),
    ("smote", SMOTE(random_state=41)),
    ("model", model)
])



# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=41, stratify=y
)

# ============================================================
# TRAIN MODEL
# ============================================================

pipeline.fit(X_train, y_train)

# ============================================================
# PREDICT + EVALUATE
# ============================================================

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nROC AUC:", roc_auc_score(y_test, y_proba))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


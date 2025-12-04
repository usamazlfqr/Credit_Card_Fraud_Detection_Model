

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
from sklearn.inspection import permutation_importance



# ============================================================
#  LOAD DATA
# ============================================================

file = "Dataset/credit-card-fraud-detection.csv"
df = pd.read_csv(file)

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
    ("smote", SMOTE(random_state=40)),
    ("model", model)
])



# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=40, stratify=y
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


# ============================================================
# GET FEATURE NAMES AFTER PREPROCESSING
# ============================================================

# Extract encoded categorical feature names
ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"]["onehot"]
ohe_features = ohe.get_feature_names_out(categorical_features)

# Full feature list in transformed space
all_features = list(numeric_features) + list(ohe_features)


# ============================================================
# PERMUTATION-BASED FEATURE IMPORTANCE (KNN)
# ============================================================


# Convert sparse matrix to dense array
X_test_transformed = pipeline.named_steps["preprocess"].transform(X_test).toarray()

results = permutation_importance(
    estimator=pipeline.named_steps["model"],
    X=X_test_transformed,
    y=y_test,
    n_repeats=3,
    random_state=40,
    scoring="f1"   # You can use recall, accuracy, etc.
)

# Store results in a DataFrame
perm_importance = pd.DataFrame({
    "feature": all_features,
    "importance_mean": results.importances_mean,
    "importance_std": results.importances_std
}).sort_values(by="importance_mean", ascending=False)


# ============================================================
# PRINT TOP IMPORTANT FEATURES
# ============================================================

print("\nPermutation-Based Feature Importance (KNN):")
print(perm_importance.head(5))

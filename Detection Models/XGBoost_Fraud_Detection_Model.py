# ============================================================
# 1. IMPORTS
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("C:/Users/User.DESKTOP-LUMR9IQ/Documents/14570063-ML-Assignment/credit-card-fraud-detection.csv")

# ============================================================
# CLEANING AND FEATURE ENGINEERING
# ============================================================

df["transaction_datetime"] = pd.to_datetime(df["transaction_datetime"], errors="coerce")
df["cardholder_age"] = pd.to_numeric(df["cardholder_age"], errors="coerce")
df["transaction_amount"] = pd.to_numeric(df["transaction_amount"], errors="coerce")

bool_cols = ["is_fraud", "is_international"]
for col in bool_cols:
    df[col] = df[col].astype(int)

df["transaction_state"] = df["transaction_state"].fillna("Intl")
df = df.dropna(subset=["transaction_city"])

id_cols = ["transaction_id", "card_number", "cardholder_id", "merchant_id"]
df = df.drop(columns=id_cols)
df = df.drop_duplicates()

# Feature engineering
df["year"] = df["transaction_datetime"].dt.year
df["month"] = df["transaction_datetime"].dt.month
df["day"] = df["transaction_datetime"].dt.day
df["hour"] = df["transaction_datetime"].dt.hour
df["dayofweek"] = df["transaction_datetime"].dt.dayofweek

# ============================================================
# SPLIT FEATURES / TARGET
# ============================================================

y = df["is_fraud"]
X = df.drop(columns=["is_fraud"])

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

# ============================================================
# PREPROCESSING PIPELINE
# ============================================================

numeric_pipeline = Pipeline([("scaler", StandardScaler())])
categorical_pipeline = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.45, random_state=42, stratify=y
)

# ============================================================
# XGBOOST + SMOTE PIPELINE
# ============================================================

xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)

pipeline = ImbPipeline(steps=[
    ("preprocess", preprocessor),
    ("model", xgb)
])

# ============================================================
# HYPERPARAMETER TUNING
# ============================================================

param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__subsample": [0.7, 0.8, 0.9],
    "model__colsample_bytree": [0.7, 0.8, 0.9],
    "model__min_child_weight": [1, 3, 5],
    "model__gamma": [0, 1, 5],
    "model__scale_pos_weight": [len(y_train[y_train==0])/len(y_train[y_train==1])]
}

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=30,
    scoring="recall",
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

search.fit(X_train, y_train)

print("Best hyperparameters:", search.best_params_)

# ============================================================
# PREDICT + THRESHOLD ADJUSTMENT
# ============================================================

best_model = search.best_estimator_

# Predict probabilities
y_proba = best_model.predict_proba(X_test)[:,1]

# Adjust threshold for better recall
threshold = 0.3  # lower threshold to catch more frauds
y_pred = (y_proba >= threshold).astype(int)

print("\nClassification Report (Threshold={}):\n".format(threshold))
print(classification_report(y_test, y_pred))
print("\nROC AUC:", roc_auc_score(y_test, y_proba))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

ohe = best_model.named_steps["preprocess"].named_transformers_["cat"]["onehot"]
ohe_features = ohe.get_feature_names_out(categorical_features)
all_features = list(numeric_features) + list(ohe_features)

importances = best_model.named_steps["model"].feature_importances_
feat_imp = pd.Series(importances, index=all_features).sort_values(ascending=False)

print("\nTop 20 Important Features:\n")
print(feat_imp.head(20))

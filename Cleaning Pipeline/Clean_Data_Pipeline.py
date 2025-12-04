import pandas as pd


# ============================================================
#   CLEANING THE DATA
# ============================================================

file = "Dataset/credit-card-fraud-detection.csv"
df = pd.read_csv(file)


print("Dataset shape:", df.shape)
print(df.head())

print("\nInitial Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


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


# ============================================================
#  SAVE DATA
# ============================================================

print("\nInitial Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Save the DataFrame to CSV
df.to_csv("credit-card-fraud-detection.csv", index=False)

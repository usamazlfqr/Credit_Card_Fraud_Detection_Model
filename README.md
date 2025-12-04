# Credit Card Fraud Detection Model
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/usamazlfqr/Credit_Card_Fraud_Detection_Model)

This repository contains a machine learning project for detecting fraudulent credit card transactions. The project implements and compares three different classification algorithms—XGBoost, Random Forest, and K-Nearest Neighbors (KNN)—with a focus on handling imbalanced datasets and extracting interpretable insights.

## Dataset

The project uses the `credit-card-fraud-detection.csv` dataset, which contains historical credit card transactions. Key features include:
- `transaction_amount`
- `merchant_category`
- `transaction_type`
- `entry_mode`
- `transaction_city`, `transaction_state`, `transaction_country`
- `cardholder_age`, `cardholder_gender`
- `is_international`
- `is_fraud` (the target variable)

## Project Structure
```
.
├── Cleaning Pipeline/
│   └── Clean_Data_Pipeline.py  # Script for data cleaning and feature engineering
├── Dataset/
│   └── credit-card-fraud-detection.csv # Raw transaction data
└── Detection Models/
    ├── KNN_Fraud_Detection_Model.py
    ├── Random_Forest_Fraud_Detection_Model.py
    └── XGBoost_Fraud_Detection_Model.py
```

## Methodology

### 1. Data Cleaning and Feature Engineering

The `Cleaning Pipeline/Clean_Data_Pipeline.py` script performs the following preprocessing steps:
- **Data Type Conversion**: Converts `transaction_datetime` to datetime objects and ensures `transaction_amount` and `cardholder_age` are numeric.
- **Missing Value Imputation**: Fills missing `transaction_state` values with "Intl" and drops records with missing `transaction_city`.
- **Data Cleaning**: Removes duplicate records and unnecessary identifier columns (`transaction_id`, `card_number`, etc.).
- **Feature Engineering**: Extracts temporal features (`year`, `month`, `day`, `hour`) from the `transaction_datetime` column to capture time-based patterns.

### 2. Modeling

Three distinct models are trained and evaluated for the fraud detection task. Each model utilizes a preprocessing pipeline that applies `StandardScaler` to numeric features and `OneHotEncoder` to categorical features.

#### Models Implemented:
1.  **XGBoost (`XGBoost_Fraud_Detection_Model.py`)**:
    - A gradient boosting framework optimized for performance and accuracy.
    - Uses `RandomizedSearchCV` for hyperparameter tuning to maximize recall.
    - Handles class imbalance using the `scale_pos_weight` parameter.
    - Implements prediction threshold adjustment to optimize for higher recall, identifying more potential fraud cases.

2.  **Random Forest (`Random_Forest_Fraud_Detection_Model.py`)**:
    - An ensemble of decision trees that improves robustness and prevents overfitting.
    - Handles class imbalance using a combination of `SMOTE` (Synthetic Minority Over-sampling Technique) and the `class_weight="balanced"` parameter.
    - Provides model-based feature importance to identify key predictors of fraud.

3.  **K-Nearest Neighbors (`KNN_Fraud_Detection_Model.py`)**:
    - A non-parametric, distance-based classifier effective at capturing local patterns.
    - Uses `SMOTE` within an `imblearn` pipeline to address the imbalanced dataset.
    - Feature importance is calculated using `permutation_importance`, which is suitable for models like KNN that do not have a built-in importance attribute.

### 3. Evaluation

The performance of each model is assessed using a comprehensive set of metrics:
- **Classification Report**: Includes precision, recall, and F1-score for both classes.
- **ROC-AUC Score**: Measures the model's ability to distinguish between fraudulent and legitimate transactions.
- **Confusion Matrix**: Visualizes the counts of true positives, false positives, true negatives, and false negatives.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/usamazlfqr/Credit_Card_Fraud_Detection_Model.git
    cd Credit_Card_Fraud_Detection_Model
    ```

2.  **Install dependencies:**
    Ensure you have Python and the necessary libraries installed.
    ```bash
    pip install pandas scikit-learn imbalanced-learn xgboost
    ```

3.  **Run the data cleaning pipeline:**
    This will generate the cleaned CSV file in the root directory.
    ```bash
    python "Cleaning Pipeline/Clean_Data_Pipeline.py"
    ```

4.  **Run a detection model:**
    Execute any of the model scripts to train, evaluate, and see the results printed to the console.
    ```bash
    # Example for XGBoost
    python "Detection Models/XGBoost_Fraud_Detection_Model.py"

    # Example for Random Forest
    python "Detection Models/Random_Forest_Fraud_Detection_Model.py"

    # Example for KNN
    python "Detection Models/KNN_Fraud_Detection_Model.py"
    ```

## Results Summary
- **XGBoost** emerged as the best-performing model, achieving high recall and ROC-AUC scores, especially after hyperparameter tuning and threshold adjustment.
- **Random Forest** also performed well, providing strong feature importance insights that aligned with the XGBoost model.
- **KNN** offered an alternative modeling approach with interpretable results from permutation-based importance, highlighting key transactional features driving predictions.

## License

This project is released under the MIT License.
# Credit_Card_Fraud_Detection_Model

Credit Card Fraud Detection Model

This repository contains a machine learning project for detecting fraudulent credit card transactions. The project explores multiple classification algorithms, including XGBoost, Random Forest, and K-Nearest Neighbours (KNN), with a focus on handling imbalanced data and extracting interpretable insights.

Dataset

The dataset used in this project is sourced from GoMask AI Marketplace
 and contains historical credit card transactions with fraud labels. Key features include transaction amount, merchant details, location, entry mode, and cardholder information.

Project Overview

Fraud detection is a critical challenge in financial services due to the low prevalence of fraudulent transactions and the high cost of missed detections. This project implements three machine learning models:

XGBoost: A gradient-boosted tree ensemble optimised for speed, accuracy, and regularisation.

Random Forest: An ensemble of decision trees that improves robustness and handles mixed data types.

K-Nearest Neighbours (KNN): A non-parametric, distance-based classifier capturing local patterns in transaction behaviour.

The models are evaluated using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices. Threshold adjustment and SMOTE oversampling are applied to handle class imbalance.

Features

Full data cleaning and preprocessing, including handling missing values, feature scaling, and one-hot encoding of categorical features.

Feature engineering, such as extracting date-time components.

Permutation-based feature importance for KNN and model-based feature importance for tree ensembles.

Comprehensive evaluation metrics with visualisation-ready outputs.

Code modularity with pipelines for preprocessing, SMOTE oversampling, model training, and evaluation.


XGBoost achieved high recall and ROC-AUC, making it the best-performing model overall.

Random Forest performed comparably with strong feature importance insights.

KNN provided interpretable results using permutation-based importance, highlighting key transactional features.

License

This project is released under the MIT License.

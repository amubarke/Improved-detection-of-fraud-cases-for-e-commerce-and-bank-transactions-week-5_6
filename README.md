# Improved-detection-of-fraud-cases-for-e-commerce-and-bank-transactions-week-5_6

**Project:** Improved Detection of Fraud Cases  
**Company:** Adey Innovations Inc.  

---

## 1. Project Overview

This project aims to detect fraudulent transactions in e-commerce and banking using machine learning.  

Key steps include:

- Data cleaning & preprocessing  
- Feature engineering (time-based, transaction frequency/velocity, IP geolocation)  
- Handling class imbalance (SMOTE oversampling)  
- Model building: Logistic Regression baseline, ensemble models (Random Forest, XGBoost, LightGBM)  
- Explicit hyperparameter tuning using GridSearchCV / RandomizedSearchCV  
- Stratified K-Fold cross-validation for robust performance estimation  
- Model evaluation: AUC-PR, F1-score, confusion matrix  
- Model explainability using SHAP  
- Providing actionable business recommendations  

**Business Objective:**  
Detect fraudulent transactions accurately while minimizing false positives, reducing financial loss, and improving customer trust.

---

## 2. Folder Structure
fraud-detection/
├── .vscode/
│ └── settings.json
├── .github/
│ └── workflows/
│ └── unittests.yml
├── data/ # Add this folder to .gitignore
│ ├── raw/ # Original datasets
│ └── processed/ # Cleaned and feature-engineered data
├── notebooks/
│ ├── init.py
│ ├── eda-fraud-data.ipynb
│ ├── eda-creditcard.ipynb
│ ├── feature-engineering.ipynb
│ ├── modeling.ipynb
│ ├── shap-explainability.ipynb
│ └── README.md
├── src/
│ ├── init.py
│ ├── data_preprocessor.py
│ ├── baseline_model.py
│ ├── ensemble_model.py
│ ├── train_model.py
│ ├── features.py
│ ├── transform.py
│ ├── imbalance.py
│ └── geolocation.py
├── tests/
│ ├── init.py
│ └── test_models.py
├── models/ # Saved model artifacts
├── scripts/
│ ├── init.py
│ └── README.md
├── requirements.txt
├── README.md
└── .gitignore


---

## 3. Setup Instructions

```bash
# Clone the repository
git clone https://github.com/amubarke/Improved-detection-of-fraud-cases-for-e-commerce-and-bank-transactions-week-5_6.git
cd Improved-detection-of-fraud-cases-for-e-commerce-and-bank-transactions-week-5_6

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

4. Step-by-Step Usage in VSCode Notebook
4.1 Load Data

import pandas as pd

fraud_df = pd.read_csv("data/raw/Fraud_Data.csv")
geo_df = pd.read_csv("data/raw/IpAddress_to_Country.csv")

fraud_df.head()

4.2 Geolocation Integration
from src.geolocation import GeoLocator

geo = GeoLocator(fraud_df, geo_df, ip_col="ip_address")
fraud_df = geo.convert_ip_to_int()
fraud_df = geo.merge_with_geolocation()

fraud_country_stats = geo.fraud_summary_by_country(target_col="Class")
fraud_country_stats.head()

4.3 Feature Engineering
from src.features import FeatureEngineer

fe = FeatureEngineer(fraud_df)
fraud_df = fe.run_all()
fraud_df.head()

4.4 Data Transformation
from src.transform import DataTransformer

numerical_cols = ["Amount", "time_since_signup", "txn_count_1h", "txn_count_24h", "avg_txn_velocity"]
categorical_cols = ["browser", "device", "country", "source"]

transformer = DataTransformer(fraud_df, numerical_cols, categorical_cols, scaling="standard")
X_transformed = transformer.fit_transform()
X_transformed.head()

4.5 Handling Class Imbalance
from src.imbalance import ImbalanceHandler
from sklearn.model_selection import train_test_split

y = fraud_df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X_transformed, y, test_size=0.2, random_state=42, stratify=y
)

imb = ImbalanceHandler(method="smote")
X_train_bal, y_train_bal = imb.fit_resample(X_train, y_train)

5. Model Training & Hyperparameter Tuning
-Baseline: Logistic Regression
-Ensemble Models: Random Forest, XGBoost, LightGBM
Hyperparameter Tuning:
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from lightgbm import LGBMClassifier

param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(LGBMClassifier(random_state=42),
                    param_grid, scoring='f1', cv=cv, n_jobs=-1, verbose=2)

grid.fit(X_train_bal, y_train_bal)
best_model = grid.best_estimator_
print("Best params:", grid.best_params_)
print("Best F1:", grid.best_score_)

6. Model Evaluation & Selection

-Evaluate all models on AUC-PR, F1, precision, recall

-Compare models in a summary table:

| Model               | Hyperparameters | AUC-PR | F1 (fraud) | Precision | Recall | Notes    |
| ------------------- | --------------- | ------ | ---------- | --------- | ------ | -------- |
| Logistic Regression | default         | 0.321  | 0.286      | 0.182     | 0.671  | Baseline |
| Random Forest       | tuned           | 0.622  | 0.676      | 0.940     | 0.528  | Tuned    |
| XGBoost             | tuned           | 0.613  | 0.685      | 0.978     | 0.528  | Tuned    |
| LightGBM            | tuned           | 0.612  | 0.685      | 0.977     | 0.528  | Selected |
   -Selected model: LightGBM (balance of F1, precision, and recall)

   -Perform SHAP explainability analysis on the selected model for actionable insights.

7. SHAP Explainability

-Feature Importance: Extract built-in feature importance from LightGBM

-SHAP Summary Plot: Global feature impact visualization

-SHAP Force Plots: For 3 individual predictions

         -True Positive: Correctly identified fraud

         -False Positive: Legitimate transaction flagged as fraud

         -False Negative: Missed fraud

Interpretation:

-Top drivers of fraud: transaction_amount, hour_of_day, device_country, new_customer_flag, merchant_category

-Surprising findings: Small amounts may trigger fraud alerts (micro-fraud testing)

8. Business Recommendations

Based on SHAP insights:

1 Enhanced verification for high-risk hours

 -Transactions at certain hours consistently increase fraud risk.

 -Action: Apply 2FA or manual review.

2 Flag new customers with unusual patterns

  -new_customer_flag strongly drives fraud predictions.

  -Action: Temporary limits on first transactions.

3 Monitor high-risk merchant categories

  -Certain categories consistently contribute to fraud.

  -Action: Automatic alerts or additional verification.

9. Next Steps

-Continue hyperparameter tuning and cross-validation for ensemble models

-Maintain CI/CD integration for model retraining and testing

-Deploy production-ready fraud detection pipeline
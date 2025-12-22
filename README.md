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
- Model building: Logistic Regression baseline, ensemble models (Random Forest/XGBoost/LightGBM)  
- Model evaluation: AUC-PR, F1-score, confusion matrix  
- Model explainability using SHAP  
- Providing actionable business recommendations  

**Business Objective:**  
Detect fraudulent transactions accurately while minimizing false positives, reducing financial loss, and improving customer trust.

---

## 2. Folder Structure
fraud-detection/

├── .vscode/

│   └── settings.json

├── .github/

│   └── workflows/

│       └── unittests.yml

├── data/                           # Add this folder to .gitignore

│   ├── raw/                      # Original datasets

│   └── processed/         # Cleaned and feature-engineered data

├── notebooks/

│   ├── __init__.py

│   ├── eda-fraud-data.ipynb

│   ├── eda-creditcard.ipynb

│   ├── feature-engineering.ipynb

│   ├── modeling.ipynb

│   ├── shap-explainability.ipynb

│   └── README.md

├── src/

│   ├── __init__.py

├── tests/

│   ├── __init__.py

├── models/                      # Saved model artifacts

├── scripts/

│   ├── __init__.py

│   └── README.md

├── requirements.txt

├── README.md

└── .gitignore


---

## 3. Setup Instructions

```bash
# Clone the repository
git clone <(https://github.com/amubarke/Improved-detection-of-fraud-cases-for-e-commerce-and-bank-transactions-week-5_6.git)>
cd project_root

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

5. Next Steps

Train ensemble models (Random Forest, XGBoost, LightGBM)

Hyperparameter tuning and Stratified K-Fold cross-validation

Compare models and select the best based on performance metrics and interpretability

Provide actionable business recommendations using SHAP insights

Build a deployment-ready pipeline with CI/CD integration
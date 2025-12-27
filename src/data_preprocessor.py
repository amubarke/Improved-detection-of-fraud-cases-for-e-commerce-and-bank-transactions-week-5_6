# src/data_preprocessor.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    def __init__(self, df, target_col):
        self.df = df.copy()
        self.target_col = target_col
        self.label_encoders = {}

    def preprocess(self):
        df = self.df.copy()

        # 1 — Separate features/target
        y = df[self.target_col]
        X = df.drop(columns=[self.target_col])

        # 2 — Convert timestamps if needed
        for col in X.select_dtypes(include=["datetime", "datetimetz"]).columns:
            X[col] = X[col].astype("int64") // 10**9

        # 3 — Identify categorical columns
        cat_cols = X.select_dtypes(include=["object", "category"]).columns

        # 4 — Split categorical columns into:
        low_card_cols = [col for col in cat_cols if X[col].nunique() < 50]
        high_card_cols = [col for col in cat_cols if X[col].nunique() >= 50]

        # 5 — One-hot encode only low-cardinality columns
        if len(low_card_cols) > 0:
            X = pd.get_dummies(X, columns=low_card_cols, drop_first=True)

        # 6 — Label-encode high-cardinality columns (safe for tree models)
        for col in high_card_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le

        # Final clean dataset
        X[self.target_col] = y
        return X

    def split(self, test_size=0.2, random_state=42):
        df_clean = self.preprocess()

        X = df_clean.drop(columns=[self.target_col])
        y = df_clean[self.target_col]

        return train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )

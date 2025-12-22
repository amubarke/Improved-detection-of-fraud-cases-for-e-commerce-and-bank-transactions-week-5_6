import pandas as pd

class FraudDataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # ----------------------------------------------------------
    # 1. Handle Missing Values (Impute or Drop)
    # ----------------------------------------------------------
    def handle_missing(self, numeric_strategy="median", drop_threshold=0.5):
        """
        numeric_strategy: 'mean', 'median', 'mode'
        drop_threshold: drop columns with more than X% missing values
        """

        df = self.df

        # Drop columns with too many missing values
        missing_ratio = df.isna().mean()
        cols_to_drop = missing_ratio[missing_ratio > drop_threshold].index
        df = df.drop(columns=cols_to_drop)

        # Numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                if numeric_strategy == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif numeric_strategy == "median":
                    df[col] = df[col].fillna(df[col].median())
                elif numeric_strategy == "mode":
                    df[col] = df[col].fillna(df[col].mode()[0])

        # Categorical columns → mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isna().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])

        self.df = df
        return self

    # ----------------------------------------------------------
    # 2. Remove Duplicate Rows
    # ----------------------------------------------------------
    def remove_duplicates(self, keep="first"):
        """
        keep: "first", "last" or False
        """
        self.df = self.df.drop_duplicates(keep=keep)
        return self

    # ----------------------------------------------------------
    # 3. Correct Data Types (Based on Fraud_Data columns)
    # ----------------------------------------------------------
    def correct_dtypes(self):
        df = self.df

        # Convert to datetime
        date_columns = ["signup_time", "purchase_time"]
        for col in date_columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

        # Convert to category
        categorical_columns = ["device_id", "source", "browser", "sex"]
        for col in categorical_columns:
            df[col] = df[col].astype("category")

        self.df = df
        return self

    # ----------------------------------------------------------
    # 4. Return Final Cleaned DataFrame
    # ----------------------------------------------------------
    def get_clean_data(self):
        return self.df

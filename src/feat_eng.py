import pandas as pd

class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()

    # ----------------------------------------------------
    # 1. Convert time columns to datetime
    # ----------------------------------------------------
    def convert_time_columns(self, signup_col="signup_time", purchase_col="purchase_time"):
        self.df[signup_col] = pd.to_datetime(self.df[signup_col])
        self.df[purchase_col] = pd.to_datetime(self.df[purchase_col])
        return self.df

    # ----------------------------------------------------
    # 2. Extract time-based features
    # ----------------------------------------------------
    def add_time_features(self, purchase_col="purchase_time"):
        self.df["hour_of_day"] = self.df[purchase_col].dt.hour
        self.df["day_of_week"] = self.df[purchase_col].dt.dayofweek
        return self.df

    # ----------------------------------------------------
    # 3. Compute time since signup (in hours)
    # ----------------------------------------------------
    def add_time_since_signup(self, signup_col="signup_time", purchase_col="purchase_time"):
        delta = self.df[purchase_col] - self.df[signup_col]
        self.df["time_since_signup"] = delta.dt.total_seconds() / 3600
        return self.df

    # ----------------------------------------------------
    # 4. Transaction frequency & velocity features
    # ----------------------------------------------------
    def add_transaction_frequency(self, user_col="user_id", purchase_col="purchase_time"):
        # Sort by user and time
        self.df = self.df.sort_values(by=[user_col, purchase_col])

        # Rolling counts: past 1 hour and 24 hours
        self.df["txn_count_1h"] = (
            self.df.groupby(user_col)[purchase_col]
            .transform(lambda x: x.rolling("1h").count())
        )

        self.df["txn_count_24h"] = (
            self.df.groupby(user_col)[purchase_col]
            .transform(lambda x: x.rolling("24h").count())
        )

        # Velocity = transactions per hour (24h window)
        self.df["avg_txn_velocity"] = self.df["txn_count_24h"] / 24

        return self.df

    # ----------------------------------------------------
    # 5. Full pipeline (easy call)
    # ----------------------------------------------------
    def run_all(self):
        self.convert_time_columns()
        self.add_time_features()
        self.add_time_since_signup()
        self.add_transaction_frequency()
        return self.df

import pandas as pd

class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()

    # 1. Convert time columns to datetime
    def convert_time_columns(self, signup_col="signup_time", purchase_col="purchase_time"):
        self.df[signup_col] = pd.to_datetime(self.df[signup_col])
        self.df[purchase_col] = pd.to_datetime(self.df[purchase_col])
        return self.df

    # 2. Extract time-based features
    def add_time_features(self, purchase_col="purchase_time"):
        self.df["hour_of_day"] = self.df[purchase_col].dt.hour
        self.df["day_of_week"] = self.df[purchase_col].dt.dayofweek
        return self.df

    # 3. Time since signup
    def add_time_since_signup(self, signup_col="signup_time", purchase_col="purchase_time"):
        delta = self.df[purchase_col] - self.df[signup_col]
        self.df["time_since_signup"] = delta.dt.total_seconds() / 3600
        return self.df

    # 4. Transaction frequency & velocity
    def add_transaction_frequency(self, user_col="user_id", purchase_col="purchase_time"):
        self.df[purchase_col] = pd.to_datetime(self.df[purchase_col])
        self.df = self.df.sort_values([user_col, purchase_col])

        # Make timestamps unique per user to avoid rolling reindex errors
        self.df["purchase_time_adj"] = (
            self.df.groupby(user_col).cumcount().astype('timedelta64[ns]') + self.df[purchase_col]
        )

        # 1-hour rolling count
        txn_1h = (
            self.df.groupby(user_col)
            .rolling("1h", on="purchase_time_adj")["purchase_value"]
            .count()
            .reset_index(level=0, drop=True)
        )

        # 24-hour rolling count
        txn_24h = (
            self.df.groupby(user_col)
            .rolling("24h", on="purchase_time_adj")["purchase_value"]
            .count()
            .reset_index(level=0, drop=True)
        )

        # Assign safely back to original DataFrame
        self.df["txn_count_1h"] = txn_1h.values
        self.df["txn_count_24h"] = txn_24h.values
        self.df["txn_velocity_24h"] = self.df["txn_count_24h"] / 24.0

        # Cleanup
        self.df.drop("purchase_time_adj", axis=1, inplace=True)

        return self.df

    # 5. Full pipeline
    def run_all(self):
        self.convert_time_columns()
        self.add_time_features()
        self.add_time_since_signup()
        self.add_transaction_frequency()
        return self.df

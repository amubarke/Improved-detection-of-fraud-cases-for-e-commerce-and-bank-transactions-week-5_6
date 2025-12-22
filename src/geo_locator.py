import pandas as pd
import numpy as np

class GeoLocator:
    def __init__(self, fraud_df, geo_df, ip_col="ip_address"):
        self.fraud_df = fraud_df.copy()
        self.geo_df = geo_df.copy()
        self.ip_col = ip_col

    # -----------------------------------------------------
    # Convert IP address from string/float → integer
    # -----------------------------------------------------
    def convert_ip_to_int(self):
        self.fraud_df[self.ip_col] = (
            self.fraud_df[self.ip_col]
            .astype(str)
            .str.split(".")
            .str[0]
            .astype("int64")
        )
        return self.fraud_df

    # -----------------------------------------------------
    # Perform range-based lookup:
    # Join where ip_int is between lower_bound and upper_bound
    # -----------------------------------------------------
    def merge_with_geolocation(self):
        ip_col = self.ip_col

        # Create intervals using pandas IntervalIndex
        intervals = pd.IntervalIndex.from_arrays(
            self.geo_df["lower_bound_ip_address"],
            self.geo_df["upper_bound_ip_address"],
            closed="both"
        )

        # Match each IP to a country using interval lookup
        self.fraud_df["country"] = self.fraud_df[ip_col].map(
            lambda x: self.geo_df.iloc[np.where(intervals.contains(x))[0]]["country"].values[0]
            if len(np.where(intervals.contains(x))[0]) > 0
            else "Unknown"
        )

        return self.fraud_df

    # -----------------------------------------------------
    # Analyze fraud count by country
    # -----------------------------------------------------
    def fraud_summary_by_country(self, target_col="Class"):
        return (
            self.fraud_df[self.fraud_df[target_col] == 1]
            .groupby("country")
            .size()
            .sort_values(ascending=False)
        )

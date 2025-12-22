import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

class ImbalanceHandler:
    def __init__(self, method="smote", random_state=42):
        self.method = method
        self.random_state = random_state
        self.resampler = None

    # ----------------------------------------------------
    # Select resampling method
    # ----------------------------------------------------
    def _get_resampler(self):
        if self.method == "smote":
            return SMOTE(random_state=self.random_state)
        elif self.method == "undersample":
            return RandomUnderSampler(random_state=self.random_state)
        else:
            raise ValueError("method must be 'smote' or 'undersample'")

    # ----------------------------------------------------
    # Report class distribution
    # ----------------------------------------------------
    def class_distribution(self, y):
        return pd.Series(Counter(y)).sort_index()

    # ----------------------------------------------------
    # Fit & Resample (training data only)
    # ----------------------------------------------------
    def fit_resample(self, X_train, y_train):
        self.resampler = self._get_resampler()

        print("📌 BEFORE RESAMPLING:")
        print(self.class_distribution(y_train))

        X_res, y_res = self.resampler.fit_resample(X_train, y_train)

        print("\n📌 AFTER RESAMPLING:")
        print(self.class_distribution(y_res))

        return X_res, y_res

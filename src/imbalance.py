import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


class ImbalanceHandler:
    """
    Utility class to handle imbalanced datasets using SMOTE or undersampling.
    """

    def __init__(self, method="smote", random_state=42):
        """
        Parameters:
        - method (str): "smote" or "undersample"
        - random_state (int): reproducibility
        """
        self.method = method.lower()
        self.random_state = random_state
        self.resampler = None

    # ----------------------------------------------------
    # Create resampling strategy
    # ----------------------------------------------------
    def _create_resampler(self):
        if self.method == "smote":
            return SMOTE(random_state=self.random_state)
        elif self.method == "undersample":
            return RandomUnderSampler(random_state=self.random_state)
        else:
            raise ValueError("method must be either 'smote' or 'undersample'.")

    # ----------------------------------------------------
    # Print class distribution
    # ----------------------------------------------------
    @staticmethod
    def class_distribution(y):
        return pd.Series(Counter(y)).sort_index()

    # ----------------------------------------------------
    # Fit on TRAIN data only & resample
    # ----------------------------------------------------
    def fit_resample(self, X_train, y_train):
        """
        Perform resampling on training data.

        Returns:
            X_resampled, y_resampled
        """
        self.resampler = self._create_resampler()

        print("📊 BEFORE RESAMPLING:")
        print(self.class_distribution(y_train))

        # Apply SMOTE / undersampling
        X_resampled, y_resampled = self.resampler.fit_resample(X_train, y_train)

        print("\n📊 AFTER RESAMPLING:")
        print(self.class_distribution(y_resampled))

        return X_resampled, y_resampled

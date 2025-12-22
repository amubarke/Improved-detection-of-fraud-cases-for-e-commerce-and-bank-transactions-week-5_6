import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataTransformer:
    def __init__(self, df, numerical_cols=None, categorical_cols=None, scaling="standard"):
        self.df = df.copy()
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.scaling = scaling
        self.transformer = None

    # ----------------------------------------------------
    # 1. Build Scaler
    # ----------------------------------------------------
    def _get_scaler(self):
        if self.scaling == "standard":
            return StandardScaler()
        elif self.scaling == "minmax":
            return MinMaxScaler()
        else:
            raise ValueError("Scaling must be 'standard' or 'minmax'")

    # ----------------------------------------------------
    # 2. Build ColumnTransformer
    # ----------------------------------------------------
    def build_transformer(self):
        scaler = self._get_scaler()

        # Categorical Encoder
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        # Column Transformer
        self.transformer = ColumnTransformer(
            transformers=[
                ("num", scaler, self.numerical_cols),
                ("cat", encoder, self.categorical_cols)
            ],
            remainder="drop"
        )
        return self.transformer

    # ----------------------------------------------------
    # 3. Fit + Transform
    # ----------------------------------------------------
    def fit_transform(self):
        if self.transformer is None:
            self.build_transformer()

        transformed = self.transformer.fit_transform(self.df)

        # Build output feature names
        num_features = self.numerical_cols
        cat_features = self.transformer.named_transformers_["cat"].get_feature_names_out(self.categorical_cols)

        feature_names = list(num_features) + list(cat_features)

        return pd.DataFrame(transformed, columns=feature_names)

    # ----------------------------------------------------
    # 4. Transform only (for test data)
    # ----------------------------------------------------
    def transform(self, new_df):
        transformed = self.transformer.transform(new_df)

        num_features = self.numerical_cols
        cat_features = self.transformer.named_transformers_["cat"].get_feature_names_out(self.categorical_cols)

        feature_names = list(num_features) + list(cat_features)

        return pd.DataFrame(transformed, columns=feature_names)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EDA:
    def __init__(self, df, target_options=None):
        """
        df: pandas DataFrame
        target_options: list of possible target column names (default: ['Class', 'class'])
        The first existing column in the list will be used as the target.
        """
        self.df = df
        if target_options is None:
            target_options = ['Class', 'class']
        
        # Pick the first column that exists
        for col in target_options:
            if col in df.columns:
                self.target_column = col
                break
        else:
            raise ValueError(f"None of the target options {target_options} exist in the DataFrame.")

    # ----------------------
    # Univariate Analysis
    # ----------------------
    def univariate_analysis(self, columns=None):
        columns = columns or [c for c in self.df.columns if c != self.target_column]
        for col in columns:
            plt.figure(figsize=(6,4))
            if self.df[col].dtype in ['int64', 'float64']:
                sns.histplot(self.df[col], kde=True, bins=30)
                plt.title(f'Histogram of {col}')
            else:
                sns.countplot(x=self.df[col])
                plt.title(f'Countplot of {col}')
            plt.show()

    # ----------------------
    # Bivariate Analysis
    # ----------------------
    def bivariate_analysis(self, columns=None):
        columns = columns or [c for c in self.df.columns if c != self.target_column]
        for col in columns:
            plt.figure(figsize=(6,4))
            if self.df[col].dtype in ['int64', 'float64']:
                sns.boxplot(x=self.target_column, y=col, data=self.df)
                plt.title(f'{col} vs {self.target_column}')
            else:
                sns.countplot(x=col, hue=self.target_column, data=self.df)
                plt.title(f'{col} vs {self.target_column}')
            plt.show()

    # ----------------------
    # Class Distribution
    # ----------------------
    def class_distribution(self):
        counts = self.df[self.target_column].value_counts()
        print(f"Class Distribution for {self.target_column}:\n", counts)
        plt.figure(figsize=(6,4))
        sns.countplot(x=self.target_column, data=self.df)
        plt.title(f'{self.target_column} Class Distribution')
        plt.show()




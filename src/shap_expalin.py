# src/model_explainability.py

import pandas as pd
import matplotlib.pyplot as plt
import shap
import numpy as np
import seaborn as sns

class ModelExplainability:
    def __init__(self, model, X_test, y_test):
        """
        model   : Trained ensemble model (LightGBM, XGBoost, RF)
        X_test  : Test features (DataFrame)
        y_test  : True labels (Series or array)
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer.shap_values(self.X_test)

    # ------------------------------
    # Built-in feature importance
    # ------------------------------
    def plot_feature_importance(self, top_n=10):
        importances = self.model.feature_importances_
        feature_names = self.X_test.columns
        feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(top_n)

        plt.figure(figsize=(10,6))
        sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette="viridis")
        plt.title(f"Top {top_n} Feature Importances (Built-in)")
        plt.show()

    # ------------------------------
    # Global SHAP summary plots
    # ------------------------------
    def plot_shap_summary(self, top_n=10):
        # Bar plot
        shap.summary_plot(self.shap_values, self.X_test, plot_type="bar", max_display=top_n)
        # Beeswarm plot
        shap.summary_plot(self.shap_values, self.X_test, max_display=top_n)

    # ------------------------------
    # Force plot for individual predictions
    # ------------------------------
    def plot_force(self, idx, title=""):
        shap.initjs()
        shap.force_plot(
            self.explainer.expected_value,
            self.shap_values[idx,:],
            self.X_test.iloc[idx,:],
            matplotlib=True
        )
        print(title)

    # ------------------------------
    # Force plots for TP, FP, FN
    # ------------------------------
    def plot_force_examples(self):
        y_true = self.y_test.values
        preds = self.model.predict(self.X_test)

        # Indices
        tp_idx = np.where((preds==1) & (y_true==1))[0][0]
        fp_idx = np.where((preds==1) & (y_true==0))[0][0]
        fn_idx = np.where((preds==0) & (y_true==1))[0][0]

        self.plot_force(tp_idx, title="True Positive Example")
        self.plot_force(fp_idx, title="False Positive Example")
        self.plot_force(fn_idx, title="False Negative Example")

    # ------------------------------
    # Top N drivers of fraud
    # ------------------------------
    def top_drivers(self, top_n=5):
        shap_abs_mean = np.abs(self.shap_values).mean(axis=0)
        shap_importance_df = pd.DataFrame({
            'Feature': self.X_test.columns,
            'SHAP Importance': shap_abs_mean
        })
        top_features = shap_importance_df.sort_values(by='SHAP Importance', ascending=False).head(top_n)
        print(f"Top {top_n} Drivers of Fraud Predictions (SHAP):")
        print(top_features)
        return top_features

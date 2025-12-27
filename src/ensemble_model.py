from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, confusion_matrix, classification_report

class EnsembleModel:
    def __init__(self, model_type="rf"):
        if model_type == "rf":
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight="balanced",
                random_state=42
            )
        elif model_type == "xgb":
            from xgboost import XGBClassifier
            self.model = XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                eval_metric="logloss"
            )
        elif model_type == "lgbm":
            from lightgbm import LGBMClassifier
            self.model = LGBMClassifier(
                n_estimators=300,
                num_leaves=40,
                max_depth=-1
            )
        else:
            raise ValueError("Invalid model_type")

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:,1]

        auc_pr = average_precision_score(y_test, y_prob)

        return {
            "model_name": self.model.__class__.__name__,
            "auc_pr": auc_pr,
            "f1": f1_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }

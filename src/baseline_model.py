from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, f1_score

class BaselineModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=500, class_weight="balanced")

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)

        # AUC-PR
        auc_pr = average_precision_score(y_test, self.model.predict_proba(X_test)[:,1])

        return {
            "model_name": "Logistic Regression",
            "auc_pr": auc_pr,
            "f1": f1_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }

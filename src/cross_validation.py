from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, f1_score

class ModelCrossValidator:
    def __init__(self, model, k=5):
        self.model = model
        self.k = k

    def run(self, X, y):
        skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=42)

        auc_scores = []
        f1_scores = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            self.model.train(X_train, y_train)

            y_pred = self.model.model.predict(X_test)
            y_prob = self.model.model.predict_proba(X_test)[:,1]

            auc_scores.append(average_precision_score(y_test, y_prob))
            f1_scores.append(f1_score(y_test, y_pred))

        return {
            "auc_pr_mean": sum(auc_scores) / self.k,
            "auc_pr_std": (sum((x - (sum(auc_scores)/self.k))**2 for x in auc_scores)/self.k)**0.5,
            "f1_mean": sum(f1_scores) / self.k,
            "f1_std": (sum((x - (sum(f1_scores)/self.k))**2 for x in f1_scores)/self.k)**0.5
        }

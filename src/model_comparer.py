class ModelComparer:
    def __init__(self, results_list):
        self.results = results_list

    def compare(self):
        sorted_models = sorted(self.results, key=lambda x: x["auc_pr"], reverse=True)
        return sorted_models

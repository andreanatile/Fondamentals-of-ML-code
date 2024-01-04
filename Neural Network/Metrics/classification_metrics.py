import numpy as np

class ClassificationMetrics:
    def __init__(self, true_labels, predicted_labels):
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels

    def compute_errors(self):
        accuracy = self.accuracy()
        precision = self.precision()
        recall = self.recall()
        f1_score = self.f1_score()

        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score
        }

    def accuracy(self):
        correct_predictions = np.sum(self.true_labels == self.predicted_labels)
        total_samples = len(self.true_labels)
        accuracy = correct_predictions / total_samples
        return accuracy

    def precision(self):
        true_positive = np.sum((self.true_labels == 1) & (self.predicted_labels == 1))
        false_positive = np.sum((self.true_labels == 0) & (self.predicted_labels == 1))
        
        precision = true_positive / (true_positive + false_positive + 1e-10)
        return precision

    def recall(self):
        true_positive = np.sum((self.true_labels == 1) & (self.predicted_labels == 1))
        false_negative = np.sum((self.true_labels == 1) & (self.predicted_labels == 0))

        recall = true_positive / (true_positive + false_negative + 1e-10)
        return recall

    def f1_score(self):
        precision = self.precision()
        recall = self.recall()

        f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
        return f1_score

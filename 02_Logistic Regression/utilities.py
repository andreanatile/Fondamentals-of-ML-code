import numpy as np

def evaluate_accuracy(y_true, y_pred):
    """
    Evaluate the accuracy of the model.

    Parameters:
    - y_true: Array of true labels
    - y_pred: Array of predicted labels

    Returns:
    - Accuracy as a percentage
    """
    correct_predictions = np.sum((y_pred >= 0.5) == y_true)
    total_samples = len(y_true)
    accuracy = correct_predictions / total_samples * 100.0

    return accuracy

def calculate_f1_score(y_true, y_pred):
    """
    Calculate the F1 score.

    Parameters:
    - y_true: Array of true labels
    - y_pred: Array of predicted labels

    Returns:
    - F1 score
    """
    # Calculate precision, recall, and F1 score
    true_positive = np.sum((y_pred >= 0.5) & (y_true == 1))
    false_positive = np.sum((y_pred >= 0.5) & (y_true == 0))
    false_negative = np.sum((y_pred < 0.5) & (y_true == 1))

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0

    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return f1_score

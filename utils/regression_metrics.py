import numpy as np

class RegressionMetrics:
    def __init__(self, true_values, predicted_values):
        self.true_values = true_values
        self.predicted_values = predicted_values

    def mean_absolute_error(self):
        return np.mean(np.abs(self.true_values - self.predicted_values))

    def mean_squared_error(self):
        return np.mean((self.true_values - self.predicted_values) ** 2)

    def r_squared(self):
        mean_true = np.mean(self.true_values)
        ss_total = np.sum((self.true_values - mean_true) ** 2)
        ss_residual = np.sum((self.true_values - self.predicted_values) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        return r2

    def compute_errors(self):
        errors = {
            'Mean Absolute Error (MAE)': self.mean_absolute_error(),
            'Mean Squared Error (MSE)': self.mean_squared_error(),
            'R-squared (R2)': self.r_squared()
        }
        return errors

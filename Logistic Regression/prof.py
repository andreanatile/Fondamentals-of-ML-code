import numpy as np
np.random.seed(42)

class LogisticRegression:
    
    def __init__(self, learning_rate=1e-2, n_steps=2000, n_features=1):
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        if n_features is not None:
            self.theta = np.random.rand(n_features)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def fit_full_batch(self, X, y):
        # Full Batch Gradient Descent
        # - Computes gradient using the entire training dataset at each iteration.
        # - Updates model parameters using the average error over the entire dataset.
        # - Suitable for smaller datasets.
        # - Generally more stable convergence compared to other methods.
        # - Computational intensity may limit its use for large datasets.
        m = len(X)
        cost_history = np.zeros(self.n_steps)
        theta_history = np.zeros((self.n_steps, self.theta.shape[0]))
        print(self.theta)

        for step in range(0, self.n_steps):
            z = np.dot(X, self.theta)
            predictions = self.sigmoid(z)
            error = predictions - y
            self.theta = self.theta - (self.learning_rate * (1/m) * np.dot(X.T, error))
            theta_history[step, :] = self.theta.T

            # Cross-entropy loss
            cost = -1 / m * (np.dot(y, np.log(predictions)) + np.dot(1 - y, np.log(1 - predictions)))
            cost_history[step] = cost

        return cost_history, theta_history

    def predict(self, X):
        z = np.dot(X, self.theta)
        predictions = self.sigmoid(z)
        return predictions


    def fit_mini_batch(self, X, y, batch_size=8):
        # Mini Batch Gradient Descent
        # - Splits the training dataset into mini-batches and computes gradients using each mini-batch.
        # - Updates model parameters using the average error over each mini-batch.
        # - A compromise between full batch and stochastic gradient descent.
        # - Suitable for moderate-sized datasets.
        # - Enables parallelism and can be faster than full batch for large datasets.
        m = len(X)
        cost_history = np.zeros(self.n_steps)
        theta_history = np.zeros((self.n_steps, self.theta.shape[0]))

        for step in range(self.n_steps):
            total_error = np.zeros(X.shape[1])  # Initialize error accumulator
            for i in range(0, m, batch_size):
                xi = X[i:i + batch_size]
                yi = y[i:i + batch_size]
                z = np.dot(xi, self.theta)
                predictions = self.sigmoid(z)
                error = predictions - yi
                total_error += np.dot(xi.T, error)

            self.theta = self.theta - (self.learning_rate * (1 / batch_size) * total_error)
            theta_history[step, :] = self.theta.T

            z = np.dot(X, self.theta)
            predictions = self.sigmoid(z)
            cost = -1 / m * (np.dot(y, np.log(predictions)) + np.dot(1 - y, np.log(1 - predictions)))
            cost_history[step] = cost

        return cost_history, theta_history

    def fit_sgd(self, X, y):
        # Stochastic Gradient Descent
        # - Updates model parameters using a single randomly selected training example at each iteration.
        # - Fastest to converge but may have high variance in parameter updates.
        # - Suitable for large datasets as it processes one data point at a time, saving memory.
        # - May require more iterations to converge and may exhibit oscillations during training.
        m = len(X)
        cost_history = np.zeros(self.n_steps)
        theta_history = np.zeros((self.n_steps, self.theta.shape[0]))

        for step in range(self.n_steps):
            random_index = np.random.randint(m)
            xi = X[random_index]
            yi = y[random_index]
            z = np.dot(xi, self.theta)
            prediction = self.sigmoid(z)
            error = prediction - yi
            self.theta = self.theta - self.learning_rate * xi.T.dot(error)

            theta_history[step, :] = self.theta.T
            z = np.dot(X, self.theta)
            predictions = self.sigmoid(z)
            cost = -1 / m * (np.dot(y, np.log(predictions)) + np.dot(1 - y, np.log(1 - predictions)))
            cost_history[step] = cost

        return cost_history, theta_history
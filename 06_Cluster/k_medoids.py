from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

class KMedoids(object):
    def __init__(self, n_clusters=2, dist=euclidean_distances, random_state=42):
        """
        Class KMedoids for implementing the K-Medoids algorithm.

        Parameters:
        - n_clusters: Number of clusters to form (default: 2).
        - dist: Distance metric to use (default: euclidean_distances).
        - random_state: Seed for random number generation (default: 42).
        """
        self.n_clusters = n_clusters
        self.dist = dist
        self.rstate = np.random.RandomState(random_state)
        self.cluster_centers_ = []  # Contains indices of the medoids
        self.indices = []  # Contains initial indices of the medoids
        self.y_pred = None  # Contains predicted labels for samples

    def fit(self, X):
        """
        Method to train the K-Medoids algorithm on input data X.

        Parameters:
        - X: Array-like or matrix, shape (n_samples, n_features).
        """
        # Create an alias for the randint function from self.rstate
        rint = self.rstate.randint
        # Initialize the list of indices with a random index
        self.indices = [rint(X.shape[0])]  # Initializing the first medoid
        # Loop to obtain self.n_clusters - 1 unique initial indices for medoids
        for _ in range(self.n_clusters - 1):
            # Generate a random index between 0 (inclusive) and X.shape[0] (exclusive)
            i = rint(X.shape[0])
            # Ensure the generated index is unique
            while i in self.indices:
                i = rint(X.shape[0])
            # Add the unique index to the list of medoid indices
            self.indices.append(i)
        # Use the selected indices to initialize the medoids (cluster centers)
        self.cluster_centers_ = X[self.indices, :]  # Initializing the medoids

        # Calculate the initial cost and predicted labels using the current medoid indices
        cost, self.y_pred = self.compute_cost(X, self.indices)

        # Initialize variables for tracking the best configuration
        new_cost = cost
        new_y_pred = self.y_pred.copy()
        new_indices = self.indices[:]

        # Flag to control the first iteration of the while loop
        initial = True

        # Main loop for swapping medoids to improve cost
        while (new_cost < cost) | initial:
            # Update initial flag to indicate subsequent iterations
            initial = False
            # Update cost and predicted labels with the current best configuration
            cost = new_cost
            self.y_pred = new_y_pred
            self.indices = new_indices
            # Iterate over each cluster
            for k in range(self.n_clusters):
                # Find indices of data points belonging to the current cluster
                k_cluster_indices = [i for i, x in enumerate(new_y_pred == k) if x]
                # Iterate over each data point in the current cluster
                for r in k_cluster_indices:
                    # If the data point is not already a medoid
                    if r not in self.indices:
                        # Create a temporary copy of the current medoid indices
                        indices_temp = self.indices[:]
                        indices_temp[k] = r  # Temporarily swap in the current data point as the medoid
                        # Compute cost and predicted labels for the new configuration
                        new_cost_temp, y_pred_temp = self.compute_cost(X, indices_temp)
                        # If the new configuration improves the cost
                        if new_cost_temp < new_cost:
                            # Update the variables to track the new best configuration
                            new_cost = new_cost_temp
                            new_y_pred = y_pred_temp
                            new_indices = indices_temp
        # Update the final medoid indices based on the best configuration
        self.cluster_centers_ = X[self.indices, :]

    def compute_cost(self, X, indices):
        """
        Method to calculate the total cost of the current medoid configuration.

        Parameters:
        - X: Array-like or matrix, shape (n_samples, n_features).
        - indices: List of current medoid indices.

        Returns:
        - cost: Total cost of the current medoid configuration.
        - y_pred: Predicted labels for samples.
        """
        # Assign each data point to the nearest medoid and obtain initial labels
        y_pred = np.argmin(self.dist(X, X[indices, :]), axis=1)
        # Calculate the total cost of the current medoid configuration
        # by summing the distances from each point to its assigned medoid
        total_cost = np.sum(
            [
                np.sum(self.dist(X[y_pred == i], X[[indices[i]], :])) for i in set(y_pred)
            ]
        )
        # Return the total cost and the predicted labels for the samples
        return total_cost, y_pred

    def predict(self, X):
        """
        Method to predict the cluster membership for each sample in X.

        Parameters:
        - X: Array-like or matrix, shape (n_samples, n_features).

        Returns:
        - Array containing the predicted cluster indices for each sample.
        """
        return np.argmin(self.dist(X, self.cluster_centers_), axis=1)

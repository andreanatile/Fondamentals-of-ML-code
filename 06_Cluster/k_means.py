from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

class KMeans(object):
    def __init__(self, n_clusters=2, dist=euclidean_distances, random_state=42):
        """
        Class KMeans for implementing the K-Means clustering algorithm.

        Parameters:
        - n_clusters: Number of clusters to form (default: 2).
        - dist: Distance metric to use (default: euclidean_distances).
        - random_state: Seed for random number generation (default: 42).
        """
        self.n_clusters = n_clusters
        self.dist = dist
        self.rstate = np.random.RandomState(random_state)
        self.cluster_centers_ = []  # Contains the cluster centers
        self.y_pred = None  # Contains predicted labels for samples

    def fit(self, X):
        """
        Method to train the K-Means algorithm on input data X.

        Parameters:
        - X: Array-like or matrix, shape (n_samples, n_features).
        """
        rint = self.rstate.randint
        initial_indices = [rint(X.shape[0])]  # Initializing the first centroid index
        for _ in range(self.n_clusters - 1): # Loop to obtain self.n_clusters - 1 unique initial indices for centroids
            # Generate a random index between 0 (inclusive) and X.shape[0] (exclusive)
            i = rint(X.shape[0])
            # Ensure the generated index is unique
            while i in initial_indices:
                i = rint(X.shape[0])
            # Add the unique index to the list of initial_indices
            initial_indices.append(i)
        # Initializing the centroids using the selected initial indices
        self.cluster_centers_ = X[initial_indices, :]

        # Flag to control the continuation of the main clustering loop
        continue_condition = True

        # Main loop for the K-Means algorithm
        while continue_condition:
            # Save a copy of the current centroids for convergence comparison
            old_centroids = self.cluster_centers_.copy()

            # Assign data points to the nearest centroid and update self.y_pred
            self.y_pred = self.predict(X)

            # Update centroids based on the mean of data points in each cluster
            for i in set(self.y_pred):
                self.cluster_centers_[i] = np.mean(X[self.y_pred == i], axis=0)

            # Check for convergence by comparing old and new centroids
            if (old_centroids == self.cluster_centers_).all():
                # If centroids haven't changed, set continue_condition to False to exit the loop
                continue_condition = False

    def predict(self, X):
        """
        Method to predict the cluster membership for each sample in X.

        Parameters:
        - X: Array-like or matrix, shape (n_samples, n_features).

        Returns:
        - Array containing the predicted cluster indices for each sample.
        """
        return np.argmin(self.dist(X, self.cluster_centers_), axis=1)
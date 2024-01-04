import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles
from cla_neural_network import NeuralNetwork  # Assuming you have the NeuralNetwork class implemented

# Moons dataset for binary classification
X_moons, y_moons = make_moons(n_samples=100, noise=0.1)

# Circles dataset for binary classification
X_circles, y_circles = make_circles(n_samples=100, noise=0.1, factor=0.5)

# Concatenate the datasets
X_binary = np.vstack([X_moons, X_circles])
y_binary = np.hstack([y_moons, y_circles])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.2, random_state=42)

# Standardize the data (optional but can be beneficial for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Instantiate the neural network
input_size = X_train.shape[1]
hidden_size = 4  # You can adjust this based on your needs
output_size = 1  # Binary classification
nn = NeuralNetwork(layers=[input_size, hidden_size,hidden_size, output_size], epochs=700, alpha=1e-2, lmd=1)

# Train the neural network
nn.fit(X_train_scaled, y_train)

# Make predictions on the test set
predictions = nn.predict(X_test_scaled)

# Compute classification metrics
metrics = nn.compute_performance(X_test_scaled, y_test)
print("Classification Metrics:")
print(metrics)

# Plot the training loss curve
nn.plot_loss()

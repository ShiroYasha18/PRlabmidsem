import numpy as np

# Define AND gate inputs and outputs
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
y = np.array([0, 0, 0, 1])  # Target outputs

# Initialize weights and bias
weights = np.random.rand(2)  # Random initialization
bias = np.random.rand(1)
learning_rate = 0.1  # Step size
epochs = 10  # Number of iterations

# Activation function (Step function)
def step_function(x):
    return 1 if x >= 0 else 0

# Training the perceptron
for epoch in range(epochs):
    for i in range(len(X)):
        # Compute output
        linear_output = np.dot(weights, X[i]) + bias
        y_pred = step_function(linear_output)

        # Update weights and bias
        error = y[i] - y_pred
        weights += learning_rate * error * X[i]
        bias += learning_rate * error

print("Trained weights:", weights)
print("Trained bias:", bias)

# Test the trained perceptron
for i in range(len(X)):
    output = step_function(np.dot(weights, X[i]) + bias)
    print(f"Input: {X[i]}, Predicted Output: {output}")

import numpy as np
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
target = iris.target  # Species labels

# Function to calculate Shannon entropy
def shannon_entropy(column):
    values, counts = np.unique(column, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs))
    return entropy

entropy_value = shannon_entropy(target)
print(f"Shannon Entropy of Iris species: {entropy_value:.4f}")

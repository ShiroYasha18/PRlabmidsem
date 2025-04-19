import numpy as np
from collections import Counter

def shannon_entropy(data):
    counts = Counter(data)
    probs = np.array(list(counts.values())) / len(data)
    return -np.sum(probs * np.log2(probs))

# Generate random categorical data (choices: A, B, C, D)
data = np.random.choice(['A', 'B', 'C', 'D'], size=100)

# Compute entropy
entropy = shannon_entropy(data)
print("Random Data:", data)
print("Shannon Entropy:", entropy)

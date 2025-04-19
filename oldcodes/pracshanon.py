import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target  # Add target column

# Function to compute Shannon entropy
def shan(column):
    value_counts = column.value_counts(normalize=True)
    entropy = -np.sum(value_counts * np.log2(value_counts))
    return entropy

# Compute entropy for the target column
entropy_value = shan(df["target"])
print(entropy_value)

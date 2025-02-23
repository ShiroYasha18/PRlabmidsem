import pandas as pd
import numpy as np

# Load Titanic dataset (Assuming the dataset is in 'titanic.csv')
df = pd.read_csv("titanic.csv")

# Function to calculate Shannon entropy
def shannon_entropy(column):
    value_counts = column.value_counts(normalize=True)  # Get probabilities
    entropy = -np.sum(value_counts * np.log2(value_counts))  # Compute entropy
    return entropy

# Select a column to compute entropy (e.g., 'Survived' or 'Pclass')
column_name = "Survived"  # Change this to any other categorical column
entropy_value = shannon_entropy(df[column_name].dropna())  # Drop NaN values

print(f"Shannon Entropy of '{column_name}': {entropy_value:.4f}")

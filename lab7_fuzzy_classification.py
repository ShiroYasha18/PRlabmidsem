import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.datasets import load_iris

data = load_iris().data[:, 2]  # Petal length
x = np.linspace(data.min(), data.max(), 100)

low = fuzz.trimf(x, [1, 1.5, 3])
med = fuzz.trimf(x, [2.5, 3.5, 5])
high = fuzz.trimf(x, [4.5, 6, 7])

# Calculate fuzzy membership values for each data point
low_membership = [fuzz.interp_membership(x, low, val) for val in data]
med_membership = [fuzz.interp_membership(x, med, val) for val in data]
high_membership = [fuzz.interp_membership(x, high, val) for val in data]

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(data, low_membership, label='Low', color='blue', alpha=0.6, edgecolor='k')
plt.scatter(data, med_membership, label='Medium', color='green', alpha=0.6, edgecolor='k')
plt.scatter(data, high_membership, label='High', color='red', alpha=0.6, edgecolor='k')
plt.xlabel("Petal Length")
plt.ylabel("Membership Value")
plt.title("Fuzzy Classification of Petal Length")
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

import numpy as np

from perc import learning_rate

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
y = np.array([0, 0, 0, 1])

weights= np.random.rand(1)
bias=np.random.rand(2)
epochs=10
learning_rate=0.1

def step(x):
    return 1 if x>=0 else 0

for i in range(epochs):
    for j in range(len(X)):
        linear_out=np.dot(X,weights)+bias
        y_pred= step(linear_out)

        error= (y_pred-y)
        weights += learning_rate*error*X[i]
        bias += learning_rate*error

print(weights)
print(bias)

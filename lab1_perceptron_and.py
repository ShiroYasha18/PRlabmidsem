import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0, 0, 0, 1])
w = np.zeros(2)
b = 0
lr = 1

for _ in range(10):
    for i in range(4):
        y_pred = 1 if np.dot(X[i], w) + b > 0 else 0
        w += lr * (y[i] - y_pred) * X[i]
        b += lr * (y[i] - y_pred)

print("Weights:", w, "Bias:", b)
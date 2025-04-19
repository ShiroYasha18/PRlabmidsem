import numpy as np

from perc_practice import y_pred

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
y = np.array([0, 0, 0, 1])

weights=np.random.rand(1)
bias= np.random.rand(2)
learning_rate= 0.1
epoch =10
def step(x):
    return 1 if x>=0 else 0
for i in range(epoch):
    for j in range(len(X)):
        out= weights*X[j]+bias
        y_pred =step(out)

        error= y_pred-y[j]
        weights+= error*learning_rate*X[i]
        bias += error*learning_rate*y[j]

print(weights)
print (bias)




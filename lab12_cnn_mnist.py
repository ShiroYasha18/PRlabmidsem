import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import numpy as np 
# Load and prepare data
digits = load_digits()
X = digits.images[..., np.newaxis] / 16.0
y = tf.keras.utils.to_categorical(digits.target, 10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build simple CNN
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(8, 8, 1)),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, verbose=0)

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("CNN Test Accuracy:", acc)

from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report,accuracy_score

# Load dataset
X, y = load_iris(return_X_y=True)

# Train classifier
model = GaussianNB()
model.fit(X, y)

# Predict
pred = model.predict(X)

# Print classification report
print(classification_report(y, pred))
print(accuracy_score(y, pred))

import numpy as np
from perceptron import Perceptron
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
X, y = load_digits(return_X_y=True)
# Normalize features
X = X / 16.0  

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Parameters
rate = 0.1
epoch = 100

# OvA training
# Create perceptron for every digit
classifiers = {}
for digit in range(10):
    print(f"Training perceptron for digit {digit}...")
    # Create binary labels for the current digit
    y_digit = (y_train == digit).astype(int)
    perceptron = Perceptron(alpha=rate)
    #sklearn_perceptron = SklearnPerceptron(alpha=rate, max_iter=epoch, tol=None, random_state=42)
    perceptron.train(X_train, y_digit, epochs=epoch)
    #sklearn_perceptron.fit(X_train, y_digit)
    classifiers[digit] = perceptron

# Prediction
def predict(X, classifiers):
    # Get scores for each classifier
    scores = np.array([clf.predict(X) for clf in classifiers.values()])
    return np.argmax(scores, axis=0)

# Evaluate model
y_pred = predict(X_test, classifiers)
#y_pred_sklearn = sklearn_perceptron.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

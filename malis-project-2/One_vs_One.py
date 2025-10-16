import numpy as np
from perceptron import Perceptron
from itertools import combinations
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
# Load dataset
X, y = load_digits(return_X_y=True)
# Normalize features
X = X / 16.0  

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Parameters
rate = 0.01
epoch = 100

# create perceptron for every pair
pairs = list(combinations(range(10), 2))
classifiers = {}

# Training perceptrons
for pair in pairs:
    print(f"Training perceptron for digit pair {pair}...")
    # Subset of the dataset for the pair
    y_pair = y_train[np.isin(y_train, pair)]  
    X_pair = X_train[np.isin(y_train, pair)]
    # Binary labels for the pair
    y_binary = np.where(y_pair == pair[0], 1, -1)  
    perceptron = Perceptron(max_iter=100)
    perceptron.fit(X_pair, y_binary)
#    perceptron = Perceptron(alpha=rate)
#    perceptron.train(X_pair, y_binary, epochs=epoch)
    classifiers[pair] = perceptron

# Prediction
def predict(X, classifiers):
    #vote for every digit
    votes = np.zeros((X.shape[0], 10))
    for (digit1, digit2), clf in classifiers.items():
        preds = clf.predict(X)
        votes[:, digit1] += (preds == 1)
        votes[:, digit2] += (preds == -1)
    return np.argmax(votes, axis=1)

# Evaluation
#y_pred = predict(X_train, classifiers)
y_pred = predict(X_test, classifiers)
#print("Accuracy:", accuracy_score(y_train, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
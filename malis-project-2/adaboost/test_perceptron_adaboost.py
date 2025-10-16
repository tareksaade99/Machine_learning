import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from adaboost import SAMME
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, classification_report
from perceptron import Perceptron
#from sklearn.linear_model import Perceptron


# Load dataset
X, y = load_digits(return_X_y=True)
# Normalize features
X = X / 16.0 

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define weak learners (perceptrons)
num_weak_learners = 1
weak_learners = [Perceptron() for _ in range(num_weak_learners)]

# Train weak learners independently
for learner in weak_learners:
    w, b = learner.train(X_train, y_train,learning_rate=0.05, n_iters=500)


''''
for idx, perc in enumerate(weak_learners):
    y_pred = perc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Perceptron {idx + 1}:")
    print(f"  Accuracy: {acc:.4f}")
'''''

# Initialize SAMME
num_classes = 10
adaboost = SAMME(num_weak_learners, num_classes)

# Train SAMME
train_data = [(X_train[i], y_train[i]) for i in range(len(y_train))]
adaboost.train(train_data, weak_learners)
#adaboost.visualize_performance()
# Predict and evaluate
y_pred = adaboost.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy * 100:.2f}%")

import numpy as np
from perceptron import Perceptron
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import Perceptron as SklearnPerceptron
import time

custom_rate = 0.001
custom_epochs = 100
sk_rate = 0.001
sk_epochs = 50

# Generate a more challenging dataset
def generate_complex_data(n_samples=1000, n_features=20, noise_level=0.1):
    """
    Generates a dataset with:
    - n_samples: number of data points
    - n_features: number of features per sample
    - noise_level: proportion of labels randomly flipped to introduce noise
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(0.8 * n_features),  # 80% of features are informative
        n_redundant=int(0.2 * n_features),    # 20% are redundant
        n_clusters_per_class=2,
        flip_y=noise_level,                   # Introduce label noise
        class_sep=1.0,                        # Reduce class separation for difficulty
        random_state=42
    )
    # Convert labels from (0, 1) to (-1, 1) for your implementation
    y_custom = np.where(y == 0, -1, 1)
    return X, y, y_custom

# Test the custom Perceptron and sklearn Perceptron
def test_comparison():
    # Generate challenging dataset
    X, y_sklearn, y_custom = generate_complex_data(n_samples=1000, n_features=20, noise_level=0.15)
    
    # Split into training and testing sets
    X_train, X_test, y_train_sklearn, y_test_sklearn = train_test_split(X, y_sklearn, test_size=0.3, random_state=42)
    _, _, y_train_custom, y_test_custom = train_test_split(X, y_custom, test_size=0.3, random_state=42)

    # Custom Perceptron Implementation
    print("Training Custom Perceptron...")
    custom_perceptron = Perceptron(alpha= custom_rate)
    start_time = time.time()
    custom_perceptron.train(X_train, y_train_custom, epochs= custom_epochs)
    custom_training_time = time.time() - start_time

    y_pred_custom = custom_perceptron.predict(X_test)
    custom_accuracy = accuracy_score(y_test_custom, y_pred_custom)
    print(f"Custom Perceptron Accuracy: {custom_accuracy:.4f}")
    print(f"Custom Perceptron Training Time: {custom_training_time:.4f} seconds")

    # Sklearn Perceptron Implementation
    print("\nTraining Sklearn Perceptron...")
    sklearn_perceptron = SklearnPerceptron(alpha= sk_rate, max_iter= sk_epochs, tol=None, random_state=42)
    start_time = time.time()
    sklearn_perceptron.fit(X_train, y_train_sklearn)
    sklearn_training_time = time.time() - start_time

    y_pred_sklearn = sklearn_perceptron.predict(X_test)
    sklearn_accuracy = accuracy_score(y_test_sklearn, y_pred_sklearn)
    print(f"Sklearn Perceptron Accuracy: {sklearn_accuracy:.4f}")
    print(f"Sklearn Perceptron Training Time: {sklearn_training_time:.4f} seconds")

    # Compare Results
    print("\nComparison of Results:")
    print(f"Custom Perceptron vs Sklearn Perceptron")
    print(f"Accuracy: {custom_accuracy:.4f} vs {sklearn_accuracy:.4f}")
    print(f"Training Time: {custom_training_time:.4f} seconds vs {sklearn_training_time:.4f} seconds")

# Run the comparison
test_comparison()

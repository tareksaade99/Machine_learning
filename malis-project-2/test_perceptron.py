import numpy as np
import perceptron as p
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a linearly separable dataset
def generate_data():
    X, y = make_classification(
        n_samples=200,       # Number of samples
        n_features=2,        # Number of features
        n_informative=2,     # Number of informative features
        n_redundant=0,       # Number of redundant features
        n_clusters_per_class=1,
        class_sep=2.0,       # Separation between classes
        random_state=42
    )
    # Convert labels from (0, 1) to (-1, 1) for the perceptron
    y = np.where(y == 0, -1, 1)
    return X, y

# Test the Perceptron implementation
def test_perceptron():
    # Generate data
    X, y = generate_data()
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the perceptron
    perceptron = p.Perceptron(alpha=0.01)
    perceptron.train(X_train, y_train, epochs=20)

    # Predict on test data
    y_pred = perceptron.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Plot decision boundary (optional visualization)
    try:
        import matplotlib.pyplot as plt
        
        # Decision boundary plotting
        def plot_decision_boundary(perceptron, X, y):
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
            grid = np.c_[xx.ravel(), yy.ravel()]
            Z = perceptron.predict(grid)
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
            plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap=plt.cm.Paired)
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.title("Perceptron Decision Boundary")
            plt.show()
        
        # Plot training results
        #plot_decision_boundary(perceptron, X_train, y_train)
        plot_decision_boundary(perceptron, X_test, y_test)
    except ImportError:
        print("Matplotlib is not installed. Skipping visualization.")

# Run the test
test_perceptron()

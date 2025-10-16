import numpy as np
import matplotlib.pyplot as plt


class SAMME:
    """
    SAMME - Multi-class AdaBoost algorithm
    """

    def __init__(self, num_learner: int, num_cats: int):
        if num_cats < 2:
            raise ValueError(f"num_cats should be at least 2, but got {num_cats}")
        self.num_learner = num_learner
        self.num_cats = num_cats
        self.entry_weights = None
        self.learner_weights = None
        self.sorted_learners = None

    def train(self, train_data: list, learners: list):
        """
        Train the AdaBoost model.
        :param train_data: List of (features, label) tuples.
        :param learners: List of weak learner objects with a `predict(X)` method.
        """
        print("Starting training...")
        n = len(train_data)
        m = len(learners)

        self.entry_weights = np.full(n, 1 / n, dtype=np.float32)
        self.learner_weights = np.zeros(m, dtype=np.float32)
        self.performance_metrics = []  # Store error rates for visualization

        errors = []
        for learnr_idx, learner in enumerate(learners):
            error = 0
            for X, label in train_data:
                if learner.predict(X.reshape(1, -1)) != label:
                    error += 1
            errors.append((learner, error))

        # Sort learners by error
        self.sorted_learners = [learner for learner, _ in sorted(errors, key=lambda x: x[1])]


        # Boost each learner
        for idx, learner in enumerate(self.sorted_learners):
            # Compute weighted error
            is_wrong = np.zeros((n,))
            for entry_idx, entry in enumerate(train_data):
                X, label = entry[0], int(entry[1])
                predicted_cat = learner.predict(X.reshape(1, -1))
                if predicted_cat != label:
                    is_wrong[entry_idx] = 1
            
            # Clamp weighted_error to avoid invalid values
            weighted_error = np.sum(is_wrong * self.entry_weights) / self.entry_weights.sum()
            weighted_error = max(1e-6, min(1 - 1e-6, weighted_error))

            self.performance_metrics.append(weighted_error)
                        
            # Compute alpha (learner weight)
            self.learner_weights[idx] = max(0, np.log((1 - weighted_error) / weighted_error) + np.log(self.num_cats - 1))
            
            # Update entry weights
            is_wrong = is_wrong.flatten()
            self.entry_weights *= np.exp(self.learner_weights[idx] * is_wrong)
            self.entry_weights /= self.entry_weights.sum()  # Normalize

        self.learner_weights /= np.sum(self.learner_weights)
        print("Training completed.")

    def predict(self, data):
        """
        Predict the label for each sample in data.
        :param data: List or array of features.
        :return: Predicted class labels.
        """
        pooled_predictions = np.zeros((len(data), self.num_cats), dtype=np.float32)
        for idx, learner in enumerate(self.sorted_learners):
            predictions = np.array([learner.predict(X.reshape(1, -1)) for X in data])
            for i, pred in enumerate(predictions):
                pooled_predictions[i, pred] += self.learner_weights[idx]
        return np.argmax(pooled_predictions, axis=1)


    def visualize_performance(self):
        """
        Visualize the performance of each weak learner during training.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.performance_metrics) + 1), self.performance_metrics, marker='o', label="Error Rate")
        plt.title("Performance of Weak Learners During Training")
        plt.xlabel("Iteration")
        plt.ylabel("Error Rate")
        plt.legend()
        plt.grid(True)
        plt.show()
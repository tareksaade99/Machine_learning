# Machine Learning Course Projects

This repository contains three projects from a machine learning course, each exploring different machine learning algorithm.

## 1. Linear Regression for Classification
- **Objective:** Explore the use of linear regression for binary and multi-class classification.  
- **Datasets:** Iris dataset (binary: Setosa vs Versicolor; multi-class: Setosa, Versicolor, Virginica).  
- **Approach:**  
  - Binary classifier: Linear regression with a threshold of 0.5.  
  - Multi-class classifier: Linear regression with two thresholds, tuned for optimal accuracy.  
- **Results:**  
  - Binary classification achieved 100% accuracy.  
  - Multi-class classification also reached 100% accuracy after threshold tuning.  
- **Insights:** Linear regression works well when classes are linearly separable but is limited for non-linear data.

## 2. Perceptron Algorithm for Digit Classification
- **Objective:** Implement and understand the perceptron algorithm, and explore multi-class extensions.  
- **Datasets:** MNIST (8x8 images of handwritten digits 0–9).  
- **Approach:**  
  - Custom perceptron trained with stochastic gradient descent.  
  - Multi-class handled with One-vs-All (OvA) and One-vs-One (OvO) strategies.  
- **Results:**  
  - Binary classification achieved 97.5% accuracy.  
  - OvA for multi-class performed poorly (≈9%).  
  - OvO strategy achieved 100% on training and 97% on testing data.  
- **Insights:** Multi-class classification benefits from multiple specialized perceptrons; OvO is more effective than OvA for digits.

## 3. Multi-Class AdaBoost with Perceptrons
- **Objective:** Extend the perceptron digit classifier using AdaBoost for multi-class classification.  
- **Approach:**  
  - SAMME algorithm used to combine multiple perceptron weak learners.  
  - Weak learners trained on shuffled subsets with varying learning rates.  
  - Models validated to select optimal number of learners and training size.  
- **Results:**  
  - Boosting improved weak learners with ~60% accuracy to final performance around 90%.  
- **Insights:**  
  - Diversity among weak learners is crucial.  
  - Optimal allocation of training data and tuning the number of learners are key to boosting success.

## 4. Model Selection and Regularization with Logistic Regression

- **Objective:** Learn model selection while understanding the effects of regularization (L1 vs L2) using the Digits dataset.  
- **Dataset:** MNIST-like digits dataset from scikit-learn. The task is to classify small digits (0–4) vs large digits (5–9).  
- **Data Preparation:**  
  - Features scaled using `StandardScaler` (Z-score normalization) to ensure equal contribution of each feature.  
  - Labels were converted to binary: 0 for small digits, 1 for large digits.  
  - Data split into 80% training and 20% testing to estimate generalization error.  

- **Logistic Regression and Regularization:**  
  - L2 regularization: `penalty="l2"`, solver `"saga"`, tolerance `0.01`.  
  - L1 regularization: `penalty="l1"`, same solver and tolerance.  
  - Hyperparameter `C` (inverse of lambda) was tuned using 5-fold cross-validation over the values `[0.001, 0.01, 0.1, 1, 10, 100]`.  

- **Results:**  
  - Best L2 model achieved high accuracy on the test set.  
  - Best L1 model also achieved high accuracy, with the added benefit of sparsity in coefficients.  

- **Analysis of Regularization Effects:**  
  - **L1 Regularization:** Encourages sparsity by setting many coefficients to zero, effectively performing feature selection.  
  - **L2 Regularization:** Penalizes large coefficients without zeroing them, resulting in smoother coefficient distributions.  
  - Observed sparsity decreased with L2 and remained high for L1. L1 produced simpler models while L2 retained all features but reduced their influence.  

- **Insights:**  
  - Regularization type and strength strongly affect model interpretability and generalization.  
  - Model selection via cross-validation is crucial to identify the optimal regularization parameter.  
  - L1 is useful when feature selection or sparsity is desired, while L2 is better for retaining all features and reducing overfitting.

---

These projects collectively illustrate fundamental concepts in machine learning, from linear models to perceptron learning and ensemble methods, highlighting both strengths and limitations of different approaches in classification tasks.

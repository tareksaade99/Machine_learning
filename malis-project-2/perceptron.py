import numpy as np

class Perceptron:
    '''
    perceptron algorithm class
    __init__() initialize the model
    train() trains the model
    predict() predict the class for a new sample
    '''

    def __init__(self, alpha = 0.01):
        '''
        INPUT :
        - alpha : is a float number bigger than 0 representing the learning rate
        '''

        if alpha <= 0:
            raise Exception("Learning rate should be positive")
            
        # Weights will be initialized at the start of training
        self.weights = None

        # Initialize bias to 0
        self.bias = 0 

        # Set the learning rate
        self.alpha = alpha
        
    def adaptive_lr(self,epoch, base_lr= 0.01, decay=0.01):
        '''
        INPUT :
        - epoch : max number of iterations
        - base_lr : initial learning rate
        - decay : positive integer representing the intensity in which the rate will decay at each iteration

        OUTPUT :
        - new_rate : learning rate after decay

        ''' 
        return base_lr / (1 + decay * epoch)
    
    def train(self,X,y,epochs = 10):
        '''
        INPUT :
        - X : is a NxD numpy array containing the input features, where D is the dimension (or number of features)
        - y : is a Nx1 numpy array containing the labels for the corrisponding row of X (1 and -1 are used to represent the 2 classes)
        - epochs: is an integer bigger than zero representing the max number of iterations over all the training data set

        ''' 
        # Raise error is input if empty
        if X.size == 0 or y.size == 0:
            raise ValueError("Input data X and y cannot be empty.")
        # Raise error if input is inconsistent
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must match.")
        # Raise error if epoch <= 0 
        if epochs <= 0:
            raise Exception("Number of epochs should be positive")
        

        # Number of features in X
        n_features = X.shape[1]
        # Initialize weights to zeros
        self.weights = np.zeros(n_features)  

        for epoch in range(epochs):
            
            rate = self.alpha

            #uncomment to use adaptive rate
            #rate = self.adaptive_lr(epoch,self.alpha)
            # Iterate over each sample
            for i in range(len(X)):

                # Calculate the prediction
                z = np.dot(self.weights, X[i]) + self.bias
                y_pred = 1 if z >= 0 else -1

                # Check for misclassified points
                if (y[i]*y_pred < 0):

                    # Update weights and bias for misclassified points using SGD
                    self.weights += rate * y[i]* X[i]
                    self.bias += rate * y[i]
       
    def predict(self,X_new):

        '''
        INPUT :
        - X_new : is a MxD numpy array containing the features of new samples whose label has to be predicted

        OUTPUT :
        - y_hat : is a Mx1 numpy array containing the predicted labels for the X_new samples

        ''' 
            
        if self.weights is None:
            raise ValueError("The perceptron is not trained yet. Train it before predicting.")
        
        y_hat = []

        for x in X_new:
            z = np.dot(self.weights, x) + self.bias
            y_pred = 1 if z >= 0 else -1
            y_hat.append(y_pred)

        return np.array(y_hat)
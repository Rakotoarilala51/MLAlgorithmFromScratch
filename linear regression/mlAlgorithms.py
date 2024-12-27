import numpy as np
class LinearRegression:
    def __init__(self, lr= 0.001, n_iters=1000):
        self.lr=lr
        self.n_iters=n_iters
        self.weight=None
        self.bias=None
        
    def fit(self, X,y):
        n_samples, n_features=X.shape
        self.weight=np.zeros(n_features)
        self.bias=0
        
        for i in range(self.n_iters):
            y_pred=np.dot(X,self.weight)+self.bias
    
            dw=(1/n_samples)*np.dot(X.T, (y_pred-y))
            db=(1/n_samples)*np.sum(y_pred-y)
    
            self.weight=self.weight-self.lr*dw
            self.bias=self.bias-self.lr*db

    def predict(self, X):
        y_pred=np.dot(X, self.weight)+self.bias
        return y_pred
class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr=lr
        self.n_iters=n_iters
        self.weight=None
        self.bias=None
        
    def sigmoid(self, X):
        return 1/(1+np.exp(-X))
        
    def fit(self, X,y):
        n_samples, n_features=X.shape
        self.weight=np.zeros(n_features)
        self.bias=0
        for _ in range(self.n_iters):
            linear_pred=np.dot(X, self.weight)+self.bias
            prediction=self.sigmoid(linear_pred)
            dw=(1/n_samples)*np.dot(X.T, (prediction-y))
            db=(1/n_samples)*np.sum(prediction-y)

            self.weight-=self.lr*dw
            self.bias-=self.lr*db

    def predict(self, X):
        linear_pred=np.dot(X, self.weight)+self.bias
        y_pred=self.sigmoid(linear_pred)
        class_pred=np.where(y_pred<=0.5, 0,1)
        return class_pred
        
        
        
            
            
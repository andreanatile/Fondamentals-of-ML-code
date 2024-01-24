import numpy as np
np.random.seed(42)
from sklearn.base import BaseEstimator,TransformerMixin
def sigmoid(z):
    return 1/(1+np.exp(-z))


class LogisticRegression(BaseEstimator):
    def __init__(self,epochs=700,alpha=1e-2,lmd=1,n_features=1,b=-1):
        self.epochs=epochs
        self.alpha=alpha
        self.lmd=lmd
        self.n_features = n_features
        self.theta=np.random.rand(n_features)
        self.b=b
        
    def fit_full_batch(self,X,y):
        m=X.shape[0]
        theta_history=np.zeros((self.epochs,self.theta.shape[0]))
        cost_history=np.zeros(self.epochs)
        
        for step in range(self.epochs):
            z=np.dot(X,self.theta)
            pred=sigmoid(z)
            error=pred-y
            
            self.theta=self.theta-self.alpha*np.dot(X.T,error)*(1/m) -(1/m)*self.alpha*self.lmd*np.sum(self.theta)
            theta_history[step]=self.theta.T 
            
            cost=-(1/m)*(np.dot(y.T,np.log(pred))+np.dot(1-y.T,np.log(1-pred))) + (1/(2*m))*np.sum(np.square(self.theta))*self.lmd
            cost_history[step]=cost
            
        return cost_history,theta_history
    
    def fit_mini_batch(self,X,y,b=128):
        m=X.shape[0]
        theta_history=np.zeros((self.epochs,self.theta.shape[0]))
        cost_history=np.zeros(self.epochs)
        
        if b==-1:
            b=m
            
        for step in range(self.epochs):
            update=np.zeros(self.theta.shape[0])
            for i in range(0,m,b):
                xi=X[i:i+b]
                yi=y[i:i+b]
                
                z=np.dot(xi,self.theta)
                pred=sigmoid(z)
                error=pred-yi
                
                update +=np.dot(xi.T,error)
                
            self.theta=self.theta-self.alpha*update*(1/b)-self.lmd*(1/m)*self.alpha*np.sum(self.theta)
            theta_history[step]=self.theta.T 
            
            pred=sigmoid(np.dot(X,self.theta))
            cost=-(1/m)*(np.dot(y.T,np.log(pred))+np.dot(1-y.T,np.log(1-pred))) + (1/(2*m))*np.sum(np.square(self.theta))*self.lmd
            cost_history[step]=cost
            
        return cost_history,theta_history
    
    
    def fit_sgd(self,X,y):
        m=X.shape[0]
        theta_history=np.zeros((self.epochs,self.theta.shape[0]))
        cost_history=np.zeros(self.epochs)
        
        for step in range(self.epochs):
            random_index=np.random.randint(m)
            
            xi=X[random_index]
            yi=y[random_index]
            
            z=np.dot(xi,self.theta)
            pred=sigmoid(z)
            error=pred-yi
            
            self.theta=self.theta-self.alpha*np.dot(xi.T,error)-(
                self.lmd*self.alpha*(1/m)*np.sum(self.theta))
            theta_history[step]=self.theta.T 
            
            pred=sigmoid(np.dot(X,self.theta))
            cost=-(1/m)*(np.dot(y.T,np.log(pred))+np.dot(1-y.T,np.log(1-pred))) +(
                (1/(2*m))*np.sum(np.square(self.theta))*self.lmd)
            cost_history[step]=cost
            
        return cost_history,theta_history
    
    def fit(self,X,y=None):
        if self.b==-1:
            cost_h,theta_h=self.fit_full_batch(X,y)
        elif self.b==-2:
            cost_h,theta_h=self.fit_sgd(X,y)
        else:
            cost_h,theta_h=self.fit_mini_batch(X,y,self.b)
            
        return self
        
    def predict(self,X):
        pred=sigmoid(np.dot(X,self.theta))
        
        return np.round(pred)
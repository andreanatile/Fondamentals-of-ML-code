import numpy as np
from sklearn.base import BaseEstimator

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))


class NeuralNetwork(BaseEstimator):
    def __init__(self,layers,epochs=700,alpha=1e-2,lmd=1):
        self.layers=layers
        self.n_layers=len(layers)
        self.epochs=epochs
        self.alpha=alpha
        self.lmd=lmd
        
        self.w={}
        self.b={}
        self.loss=[]
        
        
    def init_parametes(self):
        for i in range(1,self.n_layers):
            self.w[i]=np.random.randn(self.layers[i],self.layers[i-1])
            self.b[i]=np.ones((self.layers[i],1))
            
    
    def forward_propagation(self,X):
        values={}
        
        for i in range(1,self.n_layers):
            if i==1:
                values['Z'+str(i)]=np.dot(self.w[i],X.T)+self.b[i]
            else:
                values['Z'+str(i)]=np.dot(self.w[i],values['A'+str(i-1)])+self.b[i]
                
            values['A'+str(i)]=sigmoid(values['Z'+str(i)])
            
        return values
    
    def compute_cost(self,values,y):
        pred=values['A'+str(self.n_layers-1)].T
        m=y.shape[0]
        
        cost=-(1/m)*(np.dot(y.T,np.log(pred))+np.dot(1-y.T,np.log(1-pred)))
        
        reg_sum=0
        for i in range(1,self.n_layers):
            reg_sum +=np.sum(np.average(self.w[i]))
        
        L2_reg=reg_sum*self.lmd*(1/(2*m))
        
        return cost+L2_reg
    
    def compute_cost_derivative(self,values,y):
        return -(np.divide(y.T,values)-np.divide(1-y.T,1-values))
    
    def backward_propagation(self,values,X,y):
        m=y.shape[0]
        param_upd={}
        dZ=None
        
        for i in range(self.n_layers-1,0,-1):
            if i==(self.n_layers-1):
                dA=self.compute_cost_derivative(values['A'+str(i)],y)
            else:
                dA=np.dot(self.w[i+1].T,dZ)
            
            dZ=np.multiply(dA,sigmoid_derivative(values['A'+str(i)]))
            
            if i==1:
                param_upd['W'+str(i)]=(1/m)*(
                    np.dot(dZ,X)+self.w[i]*self.lmd
                )
            else:
                param_upd['W'+str(i)]=(1/m)*(
                    np.dot(dZ,values['A'+str(i-1)].T)+self.lmd*self.w[i]
                )
            param_upd['B'+str(i)]=(1/m)*np.sum(dZ,axis=1,keepdims=True)
            
        return param_upd
    
    def update(self,param_upd):
        for i in range(1,self.n_layers):
            self.w[i] -=self.alpha*param_upd['W'+str(i)]
            self.b[i] -= self.alpha*param_upd['B'+str(i)]
            
    
    def fit(self,X,y):
        self.loss=[]
        self.init_parametes()
        
        for i in range(self.epochs):
            values=self.forward_propagation(X)
            param_upd=self.backward_propagation(values,X,y)
            self.update(param_upd)
            
            cost=self.compute_cost(values,y)
            self.loss.append(cost)
            
        return self
    
    def predict(self,X):
        values=self.forward_propagation(X)
        pred=values['A'+str(self.n_layers-1)].T 
        return np.round(pred)
    
    
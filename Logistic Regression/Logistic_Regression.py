import numpy as np 
np.random.seed(42)

class LogisticRegression:
    
    def __init__(self,learning_rate=1e-2,n_steps=2000,n_features=1):
        self.learning_rate=learning_rate
        self.n_steps=n_steps
        self.theta=np.random.rand(n_features)

    def sigmoid(z):
        return 1/(1+np.exp(-z))

    def fit_full_batch(self,X,y):
        m=len(X)

        # Create array used for storing the theta and cost throught the steps
        theta_history=np.zeros((self.n_steps,self.theta.shape[0]))
        cost_history=np.zeros(self.n_steps)

        for step in range(0,self.n_steps):
            z=np.dot(X,self.theta)
            pred=self.sigmoid(z)
            error=pred-y

            # Update and store theta
            self.theta=self.theta-self.learning_rate*np.dot(X.T,error)/m
            theta_history[step]=self.theta.T

            # Calculate and store the cost
            cost=-(1/m)*(np.dot(y.T,np.log(pred))+np.dot((1-y).T,np.log(1-pred)))
            cost_history[step]=cost
        
        return cost_history,theta_history
    
    def fit_mini_batch(self,X,y,b=128):
        m=len(X)

        # Create array used for storing the theta and cost throught the steps
        theta_history=np.zeros((self.n_steps,self.theta.shape[0]))
        cost_history=np.zeros(self.n_steps)

        for step in range(0,self.n_steps):
            total_error=np.zeros(self.theta.shape[0])
            for i in range(0,m,b):
                xi=X[i:i+b]
                yi=y[i:i+b]

                zi=np.dot(xi,self.theta)
                pred=self.sigmoid(zi)
                error=pred-yi

                total_error += np.dot(xi.T,error)
            
            # Avereage the error over b and update and store theta
            self.theta=self.theta-self.learning_rate*total_error/b
            theta_history[step]=self.theta.T

            # Calculate the total cost 
            pred=self.sigmoid(np.dot(X,self.theta))
            cost=(1/m)*(np.dot(y.T,np.log(pred))+np.dot((1-y).T,np.log(1-pred)))
            cost_history[step]=cost
        
        return cost_history,theta_history
    
    def fit_sgd(self,X,y):
        m=len(X)

        # Create array used for storing the theta and cost throught the steps
        theta_history=np.zeros((self.n_steps,self.theta.shape[0]))
        cost_history=np.zeros(self.n_steps)

        for step in range(0,self.n_steps):
            # Set the random index
            random_index=np.random.randint(m)
            xi=X[random_index]
            yi=y[random_index]

            # Prediction
            zi=np.dot(xi,self.theta)
            pred=self.sigmoid(zi)
            error=pred-yi

            # Update and store theta
            self.theta=self.theta-self.learning_rate*np.dot(xi.T,error)
            theta_history[step]=self.theta.T

            # Calculate the total cost 
            pred=self.sigmoid(np.dot(X,self.theta))
            cost=(1/m)*(np.dot(y.T,np.log(pred))+np.dot((1-y).T,np.log(1-pred)))
            cost_history[step]=cost
        
        return cost_history,theta_history
    
    def predict(self,X):
        z=np.dot(X,self.theta)
        return self.sigmoid(z)
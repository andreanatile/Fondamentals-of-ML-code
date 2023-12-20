import numpy as np

np.random.seed(42)

class LinearRegression:

    def __init__(self,learning_rate=1e-2,n_steps=2000,n_features=1):
        self.learning_rate=learning_rate
        self.n_steps=n_steps
        self.theta=np.random.rand(n_features)

    def fit_full_batch(self,X,y):
        m=len(X)

        # Create the numpy array where theta and cost are saved
        theta_history=np.zeros((self.n_steps,self.theta.shape[0]))
        cost_history=np.zeros(self.n_steps)

        # Update the theta for every step
        for step in range(0,self.n_steps):
            pred=np.dot(X,self.theta)   #m*n n*1
            error=pred-y                #m*1 -m*1

            # Update the theta and load in theta history
            self.theta=self.theta-self.learning_rate*np.dot(X.T,error)*(1/m) #nxm * m*1
            theta_history[step]=self.theta.T 

            # Load the cost inside the array
            cost_history[step]=1/(2*m)*np.dot(error.T,error)

        return cost_history,theta_history
    
    def fit_mini_batch(self,X,y,batch=100):
        m=len(X)

        # Create the numpy array where theta and cost are saved
        theta_history=np.zeros(self.n_steps,self.theta.shape[0])
        cost_history=np.zeros(self.n_steps)

        for step in range(0,self.n_steps):
            total_error=np.zeros(self.theta.shape[0])
            for i in range(0,m,batch):
                xi=X[i:i+batch]
                yi=y[i:i+batch]

                pred=np.dot(xi,self.theta)
                error=error-yi

                total_error +=np.dot(xi.T,error)
            
            # Average the total error between the batch
            self.theta=self.theta - self.learning_rate*total_error*(1/batch)
            theta_history[step]=self.theta.T

            # Calculate the total cost and save it
            pred=np.dot(X,self.theta)
            error=pred-y
            cost_history[step]=1/(2*m)*np.dot(error.T,error)
        
        return cost_history,theta_history
    
    def fit_sgd(self,X,y):
        m=len(X)

        # Create the numpy array where theta and cost are saved
        theta_history=np.zeros(self.n_steps,self.theta.shape[0])
        cost_history=np.zeros(self.n_steps)

        for step in range(0,self.n_steps):
            random_index=np.random.randint(m)

            xi=X[random_index]
            yi=y[random_index]

            # Calculate the error and update theta
            error=np.dot(xi,self.theta)-yi
            self.theta=self.theta-self.learning_rate*np.dot(xi.T,error)
            theta_history[step]=self.theta.T

            # Calcualte the cost over the entire training set
            pred=np.dot(X,self.theta)
            error=pred-y
            cost_history[step]=1/(2*m)*np.dot(error.T,error)

        return cost_history,theta_history
    
    def predict(self,X_test):
        return np.dot(X_test,self.theta)
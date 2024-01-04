from cla_neural_network import NeuralNetwork
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Load the diabetes dataset
diabetes=pd.read_csv('.\dataset\diabetes.csv')

# Select features and target variable
X=diabetes.drop('Outcome',axis=1).values
y=diabetes['Outcome'].values

# Split the dataset into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=True,random_state=42)

input_size=X_train.shape[1]
output_size=1
hidden_size=5
layers=[input_size,hidden_size,hidden_size,output_size]

nn=NeuralNetwork(layers=layers)

nn.fit(X_train,y_train)
performance=nn.compute_performance(X_test,y_test)

print(performance)
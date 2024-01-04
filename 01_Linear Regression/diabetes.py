import pandas as pd 
import numpy as np 
from Linear_Regression import LinearRegression

np.random.seed(42)

diabetes=pd.read_csv('.\dataset\diabetes.csv')

# Shuffle in order to avoid group bias
diabetes=diabetes.sample(frac=1).reset_index(drop=True)

# Divide X and y
X=diabetes.drop('Outcome',axis=1).values
y=diabetes['Outcome'].values


# Split the data in training and test set
train_size=round(0.8*len(X))
X_train=X[:train_size]
y_train=y[:train_size]

X_test=X[train_size:]
y_test=y[train_size:]

# zero normalization scaling only on training set
mean=X_train.mean()
std=X_train.std()

X_train_std=(X_train-mean)/std
X_test_std=(X_test-mean)/std

# fit the linear regression and predict
linear_regression=LinearRegression(1e-1,500,X.shape[1])

cost_hisotry,theta_history=linear_regression.fit_full_batch(X_train_std,y_train)
y_pred=linear_regression.predict(X_test_std)
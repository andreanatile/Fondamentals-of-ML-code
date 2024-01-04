import numpy as np 
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
np.random.seed(42)

# Load the diabetes dataset
diabetes=pd.read_csv('.\dataset\diabetes.csv')

# Select features and target variable
X=diabetes.drop('Outcome',axis=1).values
y=diabetes['Outcome'].values

# Split the dataset into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=True,random_state=42)

# Standardize the features using RobustScaler
scaler=RobustScaler()
X_train_rb=scaler.fit_transform(X_train)
X_test_rb=scaler.transform(X_test)

# Create a Logistic Regression model 
logistic_model=LogisticRegression(penalty='l2',C=0.1,random_state=42)

# Build a param grid
param_grid={
    'penalty':['l1','l2'],
    'C':[0.01,0.1,1,10,100],
    'tol':[0.001,0.0001]
}

# Create the grid search
gridSearch=GridSearchCV(logistic_model,param_grid,scoring='accuracy',cv=5,verbose=True)

# Train the grid search
gridSearch.fit(X_train_rb,y_train)

# Select the best estimator and best params
best_estimator=gridSearch.best_estimator_
best_params=gridSearch.best_params_

# Predict using the best estimator
y_pred=best_estimator.predict(X_test_rb)

# Evaluate the performance of the model
accuracy=accuracy_score(y_test,y_pred)
classification=classification_report(y_test,y_pred)
confmatrix=confusion_matrix(y_test,y_pred)

# Print the results
print(f'Accuracy score: \n {accuracy}')
print(f'Classification report: \n{classification}')
print(f'Confusion matrix: \n {confmatrix}')

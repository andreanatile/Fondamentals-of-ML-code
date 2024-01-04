import numpy as np 
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.compose import ColumnTransformer

# Load the diabetes dataset
diabetes=pd.read_csv('.\dataset\diabetes.csv')

# Select features and target variable
X=diabetes.drop('Outcome',axis=1).values
y=diabetes['Outcome'].values

# Split the dataset into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=True,random_state=42)

# list of index from 0 to X.shape[1] - 1
numeric_features=list(range(X.shape[1]))

# Impute with mean strategy and scaling
numeric_trasformers=Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler',RobustScaler())
])

"""
1) Numeric_transformer processed features replace 
the original numerical features in the pipeline, retaining only the modifications.
2) remainder = passthrough, the features not involved in the transformations 
are included in the output without undergoing any modification."""

preprocessor=ColumnTransformer(
    transformers=[
        ('num',numeric_trasformers,numeric_features)
    ],
    remainder='passthrough')

pipeline=Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('classifier',LogisticRegression(C=0.1,penalty='l2',random_state=42))
])

# Define the hyperparameter grid for grid search
param_grid={
    'classifier__C' : [0.001,0.01,0.1,1,10,100],
    'classifier__tol' : [0.0001,0.001],
    #'classifier__penalty' : ['l1' , 'l2'] 
}

# Create GridSearchCV object
gridSearch=GridSearchCV(pipeline,param_grid,cv=5,scoring='accuracy',verbose=True)


# Train the grid search
gridSearch.fit(X_train,y_train)

# Select the best estimator and best params
best_estimator=gridSearch.best_estimator_
best_params=gridSearch.best_params_

# Predict using the best estimator
y_pred=best_estimator.predict(X_test)

# Evaluate the performance of the model
accuracy=accuracy_score(y_test,y_pred)
classification=classification_report(y_test,y_pred)
confmatrix=confusion_matrix(y_test,y_pred)

# Print the results
print(f'Best params: \n{best_params}')
print(f'Accuracy score: \n {accuracy}')
print(f'Classification report: \n{classification}')
print(f'Confusion matrix: \n {confmatrix}')

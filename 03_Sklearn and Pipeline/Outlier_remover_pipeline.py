import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
np.random.seed(42)


class OutlierRemover(BaseEstimator,TransformerMixin):

    def __init__(self,factor=1.5):
        self.factor=factor
        self.upper_bound=[]
        self.lower_bound=[]

# For every feature of the dataset we calculate the outlier
    def outlier_detector(self,X):
        # Calculate quartiles      
        q1=np.percentile(X,25)
        q3=np.percentile(X,75)

        # Calculate IQR (Interquartile Range)
        iqr=q3-q1

        # Calculate lower and upper bounds to identify outliers
        self.lower_bound.append(q1-(self.factor*iqr))
        self.upper_bound.append(q3+(self.factor*iqr))

        
    
    def fit(self,X,y=None):
        # Initialize lower and upper bounds
        self.lower_bound=[]
        self.upper_bound=[]

        # Apply the outlier_detector function along axis 0 (columns)
        np.apply_along_axis(self.outlier_detector,axis=0,arr=X)

        return self
    
    def transform(self,X,y=None):
        # Copy the input array to avoid unwanted changes
        X=np.copy(X)

        # Iterate over all columns
        for i in range(X.shape[1]):
            x=X[:,i]

            # Masks to identify outliers
            lower_mask= x<self.lower_bound[i]
            upper_mask=x>self.upper_bound[i]

            # Set values that are considered outliers to NaN
            x[lower_mask|upper_mask]=np.nan

            # Assign the transformed column back to the original array
            X[:,i]=x 

        # Impute NaN values with the mean
        imputer=SimpleImputer(strategy='mean')
        X=imputer.fit_transform(X)

        return X


# Load the diabetes dataset
diabetes=pd.read_csv('.\dataset\diabetes.csv')

# Select features and target variable
X=diabetes.drop('Outcome',axis=1).values
y=diabetes['Outcome'].values

# Split the dataset into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=True,random_state=42)

# Define the pipeline with the custom OutlierRemover (list of index from 0 to X.shape[1] - 1)
numeric_feature=list(range(X.shape[1]))

# imputer fill any remaining missing values with the mean strategy
numeric_transform=Pipeline(steps=[
    ('outlier remover',OutlierRemover()),
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaler',RobustScaler())
])

"""
1) Numeric_transformer processed features replace 
the original numerical features in the pipeline, retaining only the modifications.
2) remainder = passthrough, the features not involved in the transformations 
are included in the output without undergoing any modification.
"""

preprocessor=ColumnTransformer(
    transformers=[
        ('num',numeric_transform,numeric_feature)
    ],
    remainder='passthrough'
)

# Preprocessor manages the removal of outliers, imputation, and standardization of numerical features
pipeline=Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('classifier',LogisticRegression(C=1,penalty='l2',random_state=42))
])

# Define the hyperparameter grid for grid search
param_grid={
    'classifier__C':[0.001,0.01,0.1,1,10,100],
    'classifier__tol':[0.0001,0.001]
}

# Create GridSearchCV object
grid=GridSearchCV(pipeline,param_grid,scoring='accuracy',error_score='raise',cv=5)


# Train the grid search
grid.fit(X_train,y_train)

# Select the best estimator and best params
best_estimator=grid.best_estimator_
best_params=grid.best_params_


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

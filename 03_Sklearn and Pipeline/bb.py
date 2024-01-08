import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
np.random.seed(42)

class OutlierRemover(BaseEstimator,TransformerMixin):
    def __init__(self,factor=1.5):
        self.factor=factor

        self.lower_bound=[]
        self.upper_bound=[]

    def outlier_detector(self,x):
        q1=np.percentile(x,25)
        q3=np.percentile(x,75)

        iqr=q3-q1
        self.lower_bound.append(q1-(iqr*self.factor))
        self.upper_bound.append(q3+(iqr*self.factor))

    def fit(self,X,y=None):
        self.lower_bound=[]
        self.upper_bound=[]

        np.apply_along_axis(self.outlier_detector,axis=0,arr=X)
        return self
    
    def transform(self,X,y=None):
        X=np.copy(X)
        for i in range(self.X.shape[1]):
            x=X[:,i]

            lower_mask=x<self.lower_bound[i]
            upper_mask=x>self.upper_bound[i]

            x[lower_mask|upper_mask]=np.nan

            X[:,i]=x 
        
        imputer=SimpleImputer(strategy='mean')
        X=imputer.fit_transform(X)
        return X
    

diabetes=pd.read_csv('.\dataset\diabetes.csv')

# Split in X and y 
X=diabetes.drop('Outcome',axis=1).values
y=diabetes['Outcome'].values

# Split train and test sets, and we shuffle them to avoid group bias
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,shuffle=True)

numeric_feature=list(range(X_train.shape[1]))

numeric_transform=Pipeline(steps=[
    ('Outlier remover',OutlierRemover()),
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaler',RobustScaler())
])

preprocessor=ColumnTransformer(
    transformers=[
        ('num',numeric_transform,numeric_feature)
    ],
    remainder='passthrough'
)

pipeline=Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('classifier',LogisticRegression(random_state=42))
])

param_grid={
    'classifier__C':[0.01,0.1,1,10],
    'classifier__tol':[0.001,0.0001]
}

grid=GridSearchCV(pipeline,param_grid,scoring='accuracy',cv=5)

grid.fit(X_train,y_train)

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

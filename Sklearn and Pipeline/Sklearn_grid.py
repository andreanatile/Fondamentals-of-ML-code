import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.model_selection import train_test_split,GridSearchCV
np.random.seed(42)

diabetes=pd.read_csv('.\dataset\diabetes.csv')
# Split in X and y 
X=diabetes.drop('Outcome',axis=1).values
y=diabetes['Outcome'].values

# Split train and test sets, and we shuffle them to avoid group bias
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,shuffle=True)

# Scaling using standascaler
scaler=StandardScaler()
X_train_std=scaler.fit_transform(X_train)
X_test_std=scaler.transform(X_test)

linear=LogisticRegression(penalty='l2')
param_grid={
    'C':[0.001,0.01,0.1,1,10,100],
    'penalty':['l1','l2'],
    'tol':[0.001,0.0001]
            }
grid=GridSearchCV(linear,param_grid,scoring='accuracy',cv=5,verbose=True)
grid.fit(X_train_std,y_train)
best_model=grid.best_estimator_

y_pred=best_model.predict(X_test_std)
best_param=grid.best_params_

accuracy=accuracy_score(y_test,y_pred)
classif=classification_report(y_test,y_pred)
confmatrix=confusion_matrix(y_test,y_pred)

print(f'Best params found by the GridSearch: \n{best_param}')
print(f'Accuracy of the best estimator: \n {accuracy}')
print(f'Classification report of the best estimator: \n{classif}')
print(f'Confusin matrix of the best estimator: \n {confmatrix}')

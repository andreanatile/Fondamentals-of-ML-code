import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,confusion_matrix
import pandas as pd 
from nn import NeuralNetwork

dataset=pd.read_csv('esame_01_31/alzheimer.csv')


categorical_feature=['Group', 'M/F']
factor=1.5
dataset.dropna()
for col in dataset.columns:
    if col not in categorical_feature:
        X=dataset[col].values.astype('float')
        q1=np.percentile(X,25)
        q3=np.percentile(X,75)
        
        iqr=q3-q1
        lower_bound=q1-iqr*factor
        upper_bound=q3+iqr*factor
        
        lower_mask=X<lower_bound
        upper_mask=X>upper_bound
        
        X[lower_mask|upper_mask]=np.nan
        
        dataset[col]=X
        
dataset=pd.get_dummies(dataset,columns=categorical_feature)
print(dataset.columns)
X=dataset.drop('Group_Demented',axis=1).values
y=dataset['Group_Demented'].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

impute=SimpleImputer(strategy='mean')
X_train=impute.fit_transform(X_train)
X_test=impute.transform(X_test)

scaler=StandardScaler()
X_train_std=scaler.fit_transform(X_train)
X_test_std=scaler.transform(X_test)

input_layers=X_train.shape[1]
hidden_layers1=8
hidden_layers2=4
output_layer=1
layers=[input_layers,hidden_layers1,hidden_layers2,output_layer]


nn=NeuralNetwork(layers,epochs=2000)
param_grid={
    'alpha':[1e-2,1e-3],
    'lmd':[0,1,10,100]
}

grid=GridSearchCV(nn,param_grid,scoring='accuracy',cv=5,error_score='raise')
grid.fit(X_train_std,y_train)
best_estimator=grid.best_estimator_
y_pred=best_estimator.predict(X_test_std)

print(f'Best params: \n {grid.best_params_}')
print(f'Accuracy score: \n {accuracy_score(y_test,y_pred)}')
print(f'Confusion matrix: \n {confusion_matrix(y_test,y_pred)}')
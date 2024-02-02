import pandas as pd 
import numpy as np 
np.random.seed(42)
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split,GridSearchCV
from nn import NeuralNetwork

# Load dataset with pandas
dataset=pd.read_csv('esame_01_31/alzheimer.csv')

# Save the labels of categorical valeus
categorical_feature=['Group', 'M/F']

# Check number of null values per variables and drop them
print(dataset.isnull().sum())
dataset=dataset.dropna()
factor=1.5

# Remove outlier for numerical variabels
numeric_feature=[]
for col in dataset.columns:
    if col not in categorical_feature:
        
        numeric_feature.append(col)
        X=dataset[col].values.astype('float')
        q1=np.percentile(X,25)
        q3=np.percentile(X,75)
        
        iqr=q3-q1
        
        lower_bound=q1-(iqr*factor)
        upper_bound=q3+(iqr*factor)
        
        lower_mask=X<lower_bound
        upper_mask=X>upper_bound
        
        X[upper_mask|lower_mask]=np.nan
        dataset[col]=X

# Impute the removed values of outlier 
impute=SimpleImputer(strategy='mean')
X=dataset[numeric_feature].values
X=impute.fit_transform(X)
dataset[numeric_feature]=X

# Convert categorical values in dummies variabels
dataset=pd.get_dummies(data=dataset,columns=categorical_feature)
print(dataset.columns)

# we are interested in predicting alzhaimer patient, so we focus on 'Group_demented'
X=dataset.drop('Group_Demented',axis=1).values
y=dataset['Group_Demented'].values

# Divide in train and test set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# Scale with Z score normalization
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

input_size=X.shape[1]
hidden_size1=8
hidden_size2=4
output_size=1

layers=[input_size,hidden_size1,hidden_size2,output_size]

nn=NeuralNetwork(layers,epochs=700,alpha=1e-2,lmd=1)

param_grid={
    'alpha':[1e-3,1e-2,1e-1],
    'lmd':[1,10,100]
}

grid=GridSearchCV(nn,param_grid,cv=5,scoring='accuracy',error_score='raise')
grid.fit(X_train_scaled,y_train)

best_estimator=grid.best_estimator_
y_pred=best_estimator.predict(X_test_scaled)

print(f'Best params: \n {grid.best_params_}')
print(f'Accuracy score: \n {accuracy_score(y_test,y_pred)}')
print(f'Confusion matrix: \n {confusion_matrix(y_test,y_pred)}')


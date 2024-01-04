import numpy as np 
import pandas as pd
from utilities import evaluate_accuracy,calculate_f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Logistic_Regression import LogisticRegression
from prof import LogisticRegression as Logprof
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

# Fit the logistic regression
logistic=LogisticRegression(learning_rate=1e-2,n_steps=2000,n_features=X.shape[1])
cost_hisotry,theta_history=logistic.fit_mini_batch(X_train_std,y_train,b=8)
y_pred=logistic.predict(X_test)

# Compare with the professors one
prof=Logprof(learning_rate=1e-2,n_steps=2000,n_features=X.shape[1])
cost_hisotry,theta_history=prof.fit_mini_batch(X_train_std,y_train,8)
y_pred_prof=logistic.predict(X_test)

# Calculate metrics and print them
accuracy=evaluate_accuracy(y_test,y_pred)
f1=calculate_f1_score(y_test,y_pred)

acc_prof=evaluate_accuracy(y_test,y_pred_prof)
f1_prof=calculate_f1_score(y_test,y_pred_prof)

print('mine')
print(f'Accuracy score: \n{accuracy}')
print(f'F1 score: \n{f1}')
print('\n\n --------------------prof--------------\n\n')
print(f'Accuracy score: \n{acc_prof}')
print(f'F1 score: \n{f1_prof}')


import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Logistic_Regression import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

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
y_pred=logistic.predict(X_test_std)

# Evaluate the performance of the model
accuracy=accuracy_score(y_test,y_pred)
classification=classification_report(y_test,y_pred)
confmatrix=confusion_matrix(y_test,y_pred)

# Print the results
print(f'Accuracy score: \n {accuracy}')
print(f'Classification report: \n{classification}')
print(f'Confusion matrix: \n {confmatrix}')


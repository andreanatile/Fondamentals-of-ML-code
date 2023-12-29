import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
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

# Create a logistic regression model
logistic_regression=LogisticRegression(penalty='l2',C=0.1,random_state=42)

# Train the logistic regression model on the standardized training data
logistic_regression.fit(X_train_rb,y_train)

# Make predictions on the standardized test data
y_pred=logistic_regression.predict(X_test_rb)

# Evaluate the performance of the model
accuracy=accuracy_score(y_test,y_pred)
classification=classification_report(y_test,y_pred)
confmatrix=confusion_matrix(y_test,y_pred)

# Print the results
print(f'Accuracy score: \n {accuracy}')
print(f'Classification report: \n{classification}')
print(f'Confusion matrix: \n {confmatrix}')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split,GridSearchCV
import matplotlib.pyplot as plt

# Load the dataset
diabetes=pd.read_csv('.\dataset\diabetes.csv')

#Divide features and target variable transforming them into matrices
X = diabetes.drop(['Outcome'], axis=1).values
y = diabetes['Outcome'].values

# Split the dataset into training and test sets through hold-out strategy
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)

param_grid={
    'n_estimators':list(range(10,51,10)),
    'criterion': ['entropy','gini'],
    'max_depth':list(range(5,21,5)),
    'min_samples_leaf':list(range(2,5,1))
}

grid=GridSearchCV(RandomForestClassifier(random_state=42),param_grid,cv=5,scoring='accuracy')
grid.fit(X_train,y_train)

best_model=grid.best_estimator_
best_params=grid.best_params_

y_pred=best_model.predict(X_test)

#Evaluation of the model
print(f"Accuracy TEST: {accuracy_score(y_test, y_pred)}")
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print(f'Best params: \n {grid.best_params_}')


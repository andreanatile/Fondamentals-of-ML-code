import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt 
np.random.seed(42)

# Load the dataset
diabetes=pd.read_csv('.\dataset\diabetes.csv')

#Divide features and target variable transforming them into matrices
X = diabetes.drop(['Outcome'], axis=1).values
y = diabetes['Outcome'].values

# Split the dataset into training and test sets through hold-out strategy
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)

# Set grid search
param_grid={
    'criterion':['gini','entropy'],
    'max_depth':range(4,11),
    'min_samples_leaf': range(1,6)
}

# Create the grid object
grid=GridSearchCV(DecisionTreeClassifier(random_state=42),param_grid,scoring='accuracy',cv=5)

# Train all the classifiers
grid.fit(X_train,y_train)

# Get the best model and hyperparameters
best_model=grid.best_estimator_
y_pred=best_model.predict(X_test)

#Evaluation of the model
print(f"Accuracy TEST: {accuracy_score(y_test, y_pred)}")
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print(f'Best params: \n {grid.best_params_}')

# Plot the tree
plt.figure(figsize=(12, 8))
plot_tree(best_model, feature_names=diabetes.columns.tolist(), filled=True)
plt.show()


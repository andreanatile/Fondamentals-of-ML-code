import pandas as pd 
import numpy as np 
from Linear_Regression import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)

diabetes=pd.read_csv('.\dataset\diabetes.csv')

# Shuffle in order to avoid group bias
diabetes=diabetes.sample(frac=1).reset_index(drop=True)

# Divide X and y
X=diabetes.drop('Glucose',axis=1).values
y=diabetes['Glucose'].values


# Split the data in training and test set
train_size=round(0.8*len(X))
X_train=X[:train_size]
y_train=y[:train_size]

X_test=X[train_size:]
y_test=y[train_size:]

# zero normalization scaling only on training set
mean=X_train.mean()
std=X_train.std()

X_train_std=(X_train-mean)/std
X_test_std=(X_test-mean)/std

# fit the linear regression and predict
linear_regression=LinearRegression(n_steps=700,n_features=X.shape[1],learning_rate=1e-2)

cost_hisotry,theta_history=linear_regression.fit_full_batch(X_train_std,y_train)
y_pred=linear_regression.predict(X_test_std)

# Evaluate using regression metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared (R2) Score: {r2}')
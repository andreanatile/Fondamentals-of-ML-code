import numpy as np
import pandas as pd
import torch
device='cuda' if torch.cuda.is_available() else 'cpu'
device
from torch import nn,optim
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.base import BaseEstimator

dataset=pd.read_csv('esame_01_31/alzheimer.csv')
dataset.dropna()
dataset.drop_duplicates()

print(dataset.columns)
categorical_feature=['Outcome']

# Remove outlier from numerical feature
factor=1.5
for col in dataset.columns:
    if col not in categorical_feature:
        X=dataset[col].values.astype('float')
        q1=np.percentile(X,25)
        q3=np.percentile(X,75)
        iqr=q3-q1
        
        lower_bound=q1-(iqr*factor)
        upper_bound=q3+(iqr*factor)
        
        lower_mask=X<lower_bound
        upper_mask=X>upper_bound
        
        X[lower_mask|upper_mask]=np.nan
        dataset[col]=X
        
X=dataset.drop('Outcome',axis=1).values
y=dataset['Outcome'].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# Impute missing values
imputer=SimpleImputer(strategy='mean')
X_train=imputer.fit_transform(X_train)
X_test=imputer.transform(X_test)

# Scale feature
scaler=StandardScaler()
X_train_std=scaler.fit_transform(X_train)
X_test_std=scaler.transform(X_test)


class NN(nn.Module):
    def __init__(self,input_size,hidden_size1,hidden_size2,output_size):
        super(NN,self).__init__()
        self.fc1=nn.Linear(input_size,hidden_size1)
        self.fc2=nn.Linear(hidden_size1,hidden_size2)
        self.fc3=nn.Linear(hidden_size2,output_size)
        self.sigmoid=nn.Sigmoid()
        
    def forward(self,X):
        X=self.fc1(X)
        X=self.sigmoid(X)
        
        X=self.fc2(X)
        X=self.sigmoid(X)
        
        X=self.fc3(X)
        X=self.sigmoid(X)
        
        return X
    
def compute_accuracy(y_true,y_pred):
    correct=torch.eq(y_true,y_pred).sum().item()
    acc=(correct/len(y_true))*100
    return acc

def compute_accuracy2(y_test,y_pred):
    return accuracy_score(y_test.numpy(),y_pred.numpy())

torch.manual_seed(42)
input_size=X_train_std.shape[1]
hidden_size1=8
hidden_size2=4
output_size=1

criterion=nn.BCELoss()

X_train_std=torch.from_numpy(X_train_std).float().to(device)
X_test_std=torch.from_numpy(X_test_std).float().to(device)
y_train=torch.from_numpy(y_train).float().to(device)
y_test=torch.from_numpy(y_test).float().to(device)

model=NN(input_size,hidden_size1,hidden_size2,output_size)
epochs=3000
param={}
confusion_dict={}
accuracy_dict={}

for lrg in [0.01,0.001,0.0001]:
    optimizer=optim.SGD(model.parameters(),lr=lrg)
    for epoch in range(epochs):
        model.train()
        outputs=model(X_train_std)
        outputs=outputs.squeeze()
        loss=criterion(outputs,y_train)
        outputs=torch.round(outputs).float()
        acc=compute_accuracy(y_train,outputs)
        
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        model.eval()
        
        with torch.inference_mode():
            outputs_test=model(X_test_std)
            outputs_test=outputs_test.squeeze()
            loss_test=criterion(outputs_test,y_test)
            outputs_test=torch.round(outputs_test).float()
            acc_test=compute_accuracy(y_test,output_size)
            
        if (epoch+1)==epochs:
            param[lrg]=loss_test
            conf_metrix=confusion_matrix(y_test.numpy(),outputs_test)
            accuracy=accuracy_score(y_test.numpy(),outputs_test)
            confusion_dict[lrg]=conf_metrix
            accuracy_dict[lrg]=accuracy
            

best_param=min(param,key=param.get)
print(f'Best learning rate: {best_param} \n Best test loss: {param[best_param]}')
print(f'Confusion matrix: \n {confusion_dict[best_param]}')
print(f'Accuracy score: \n {accuracy_dict[best_param]}')

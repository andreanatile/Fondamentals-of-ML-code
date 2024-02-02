import torch
from torch import nn
from torch import optim
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
import pandas as pd 
import numpy as np 
np.random.seed(42)
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split,GridSearchCV

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
X_train_std=scaler.fit_transform(X_train)
X_test_std=scaler.transform(X_test)

input_size=X.shape[1]
hidden_size1=8
hidden_size2=4
output_size=1

class SimpleNN_1(nn.Module):
    def __init__(self, input_size, hidden_size1,hidden_size2, output_size):
        super(SimpleNN_1, self).__init__() # SimpleNN is a sub-class of nn.Module
        self.fc1 = nn.Linear(input_size, hidden_size1) # First layer -> Hidden Layer
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3=nn.Linear(hidden_size2,output_size)# Hidden Layer -> Output
        self.sigmoid = nn.Sigmoid() # Activation (it is a classification task) -> We produce predictions

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        
        return x
    
    
    # Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc


torch.manual_seed(42)

input_size=X.shape[1]
hidden_size1=8
hidden_size2=4
output_size=1

model = SimpleNN_1(input_size, hidden_size1,hidden_size2, output_size).to(device)

criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
optimizer = optim.SGD(model.parameters(), lr=0.001)  # SGD optimizer
X_train_std = torch.from_numpy(X_train_std).float().to(device)
X_test_std = torch.from_numpy(X_test_std).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
y_test= torch.from_numpy(y_test).float().to(device)


epochs = 3000
for epoch in range(epochs):
  model.train() # we just inform the model that we are training it.
  outputs = model(X_train_std) # we obtain the predictions on training set
  outputs = outputs.squeeze() # we adapt prediction size to our labels
  # print(outputs)
  loss = criterion(outputs, y_train) # compute loss function
  outputs = torch.round(outputs).float() # transform predictions in labels
  acc = accuracy_fn(y_true=y_train,
                          y_pred=outputs)
  # compute loss gradients with respect to model's parameters
  loss.backward()
  # update the model parameters based on the computed gradients.
  optimizer.step()
  # In PyTorch, for example, when you perform backpropagation to compute
  # the gradients of the loss with respect to the model parameters, these
  # gradients accumulate by default through the epochs. Therefore, before
  # computing the gradients for a new batch, it's a common practice to zero
  # them using this line to avoid interference from previous iterations.
  optimizer.zero_grad()
  model.eval() # we just inform the model that we are evaluating it.
  with torch.inference_mode(): # we are doing inference: we don't need to compute gradients
    # 1. Forward pass
    test_outputs = model(X_test_std)
    test_outputs = test_outputs.squeeze()
    test_loss = criterion(test_outputs,
                        y_test)
    test_outputs = torch.round(test_outputs).float()
    # print(test_outputs)
    test_acc = accuracy_fn(y_true=y_test,
                            y_pred=test_outputs)
    # 2. Caculate loss/accuracy


  if (epoch + 1) % 20 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
import torch
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator,TransformerMixin

# Load the dataset

salaries=pd.read_csv('.\dataset\ds_salaries new.csv')

# Remove all duplicate
salaries=salaries.drop_duplicates()


class OutlierRemover(BaseEstimator,TransformerMixin):
    def __init__(self,factor=1.5):
        self.factor=factor
        self.lower_bound=[]
        self.upper_bound=[]
        
    def outlier_detector(self,X):
        q1=np.percentile(X,25)
        q3=np.percentile(X,75)
        
        iqr=q3-q1
        
        self.upper_bound.append(q3+(self.factor*iqr))
        self.lower_bound.append(q1-(self.factor*iqr))
        
    def fit(self,X):
        self.upper_bound=[]
        self.lower_bound=[]
        
        np.apply_along_axis(self.outlier_detector,axis=0,arr=X)
        
        return self

    def transform(self,X,y=None):
        X=np.copy(X)
        
        x=X
            
        lower_mask=x<self.lower_bound[0]
        upper_mask=x>self.upper_bound[0]
            
        x[lower_mask|upper_mask]=np.nan
            
        X=x 
            
        imputer=SimpleImputer(strategy='mean')
        X=imputer.fit_transform(X)
        return X
    
X=salaries.drop('salary_in_usd',axis=1).values
y=salaries['salary_in_usd'].values
y=y.astype('float')

# Outlier remover for y, and imputing the value with the strategy mean
outlier_remover=OutlierRemover(factor=1.5)
outlier_remover.fit(y)
y=outlier_remover.transform(y)

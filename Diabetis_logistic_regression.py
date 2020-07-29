#Let's start with importing necessary libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model  import Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LogisticRegression
from sklearn.model_selection import train_test_split
#from statsmodels.stats.outliers_influence import variance_inflation_factor
#from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
#import matplotlib.pyplot as plt
import seaborn as sns
#import scikitplot as skl
sns.set()

data = pd.read_csv("diabetes.csv") # Reading the Data
#data.head()

# replacing zero values with the mean of the column
data['BMI'] = data['BMI'].replace(0,data['BMI'].mean())
data['BloodPressure'] = data['BloodPressure'].replace(0,data['BloodPressure'].mean())
data['Glucose'] = data['Glucose'].replace(0,data['Glucose'].mean())
data['Insulin'] = data['Insulin'].replace(0,data['Insulin'].mean())
data['SkinThickness'] = data['SkinThickness'].replace(0,data['SkinThickness'].mean())

X = data.drop(columns = ['Outcome'])
y = data['Outcome']

scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)

x_train,x_test,y_train,y_test = train_test_split(X_scaled,y, test_size= 0.25, random_state = 355)

log_reg = LogisticRegression()

log_reg.fit(x_train,y_train)

import pickle

# Writing different model files to file
with open('modelForPrediction.pkl', 'wb') as f:
    pickle.dump(log_reg, f)

with open('standardScalar.pkl', 'wb') as f:
    pickle.dump(scalar, f)

y_pred = log_reg.predict(x_test)
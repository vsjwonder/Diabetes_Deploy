#Let's start with importing necessary libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model  import Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LogisticRegression
from sklearn.model_selection import train_test_split
#from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skl
import sklearn
sns.set()

data = pd.read_csv("diabetes.csv") # Reading the Data

#Replacing the zero-values for Blood Pressure
df1 = data.loc[data['Outcome'] == 1]
df2 = data.loc[data['Outcome'] == 0]
df1 = df1.replace({'BloodPressure':0}, np.median(df1['BloodPressure']))
df2 = df2.replace({'BloodPressure':0}, np.median(df2['BloodPressure']))
dataframe = [df1, df2]
data = pd.concat(dataframe)

#Replacing the zero-values for Blood Pressure
df1 = data.loc[data['Outcome'] == 1]
df2 = data.loc[data['Outcome'] == 0]
df1 = df1.replace({'BMI':0}, np.median(df1['BMI']))
df2 = df2.replace({'BMI':0}, np.median(df2['BMI']))
dataframe = [df1, df2]
data = pd.concat(dataframe)

#Replacing the zero-values for Blood Pressure
df1 = data.loc[data['Outcome'] == 1]
df2 = data.loc[data['Outcome'] == 0]
df1 = df1.replace({'Glucose':0}, np.median(df1['Glucose']))
df2 = df2.replace({'Glucose':0}, np.median(df2['Glucose']))
dataframe = [df1, df2]
data = pd.concat(dataframe)

#Replacing the zero-values for Blood Pressure
df1 = data.loc[data['Outcome'] == 1]
df2 = data.loc[data['Outcome'] == 0]
df1 = df1.replace({'Insulin':0}, np.median(df1['Insulin']))
df2 = df2.replace({'Insulin':0}, np.median(df2['Insulin']))
dataframe = [df1, df2]
data = pd.concat(dataframe)

#Replacing the zero-values for Blood Pressure
df1 = data.loc[data['Outcome'] == 1]
df2 = data.loc[data['Outcome'] == 0]
df1 = df1.replace({'SkinThickness':0}, np.median(df1['SkinThickness']))
df2 = df2.replace({'SkinThickness':0}, np.median(df2['SkinThickness']))
dataframe = [df1, df2]
data = pd.concat(dataframe)

X = data.drop(columns = ['Outcome'])
y = data['Outcome']

scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)

x_train,x_test,y_train,y_test = train_test_split(X_scaled,y, test_size= 0.20, random_state = 45)

from imblearn.over_sampling import SMOTE
smt = SMOTE()
x_train, y_train = smt.fit_sample(x_train, y_train)

#Logistic regression

log_reg = LogisticRegression()

log_reg.fit(x_train,y_train)

y_pred = log_reg.predict(x_test)

print('Accuracy of logistic regression on test set: {:.2f}'.format(log_reg.score(x_test, y_test)))

print('Logistic_F1 Score =',sklearn.metrics.f1_score(y_test, y_pred, average="macro"))
print('Logistic_Precision =',sklearn.metrics.precision_score(y_test, y_pred, average="macro"))
print('Logistic_Recall =',sklearn.metrics.recall_score(y_test, y_pred, average="macro"))

#Model Fitting: Support Vector Machine (Kernel: rbf)

import sklearn
from sklearn.svm import SVC
classifier_rbf = SVC(kernel = 'rbf')
classifier_rbf.fit(x_train, y_train)
SVM_y_pred = classifier_rbf.predict(x_test)
print('Accuracy of SVC (RBF) classifier on test set: {:.2f}'.format(classifier_rbf.score(x_test, y_test)))

print('SVM_F1 Score =',sklearn.metrics.f1_score(y_test, SVM_y_pred, average="macro"))
print('SVM_Precision =',sklearn.metrics.precision_score(y_test, SVM_y_pred, average="macro"))
print('SVM_Recall =',sklearn.metrics.recall_score(y_test, SVM_y_pred, average="macro"))

#Model Fitting: Random Forest We use Random Forest Classifier, with 300 trees
# (derived at after tuning the model) to fit a model on the data.

from sklearn.ensemble import RandomForestClassifier
RandomForestModel = RandomForestClassifier(n_estimators=300, bootstrap = True, max_features = 'sqrt')
RandomForestModel.fit(x_train, y_train)
RandomForest_y_pred = RandomForestModel.predict(x_test)
print('Accuracy of Random Forest on test set: {:.2f}'.format(RandomForestModel.score(x_test, y_test)))

print('RandomForest_F1 Score =',sklearn.metrics.f1_score(y_test, RandomForest_y_pred, average="macro"))
print('RandomForest_Precision =',sklearn.metrics.precision_score(y_test, RandomForest_y_pred, average="macro"))
print('RandomForest_Recall =',sklearn.metrics.recall_score(y_test, RandomForest_y_pred, average="macro"))

#Grid Search CV optimised Random forest model
#from sklearn.model_selection import GridSearchCV
#rand_clf = RandomForestClassifier(random_state=6)
# we are tuning three hyperparameters right now, we are passing the different values for both parameters
#        grid_param = {
#            "n_estimators" : [100,250,500],
#            'criterion': ['gini', 'entropy'],
#            'max_depth' : range(2,20,2),
#            'min_samples_leaf' : range(1,10,2),
#            'min_samples_split': range(2,10,2),
#            'max_features' : ['auto','log2']
#        }
# grid_search = GridSearchCV(estimator=rand_clf,param_grid=grid_param,cv=5,n_jobs =2,verbose = 3)
# grid_search.fit(x_train,y_train)
#let's see the best parameters as per our grid search
#grid_search.best_params_

rand_GSOpt = RandomForestClassifier(criterion= 'entropy',
 max_depth = 14,
 max_features = 'log2',
 min_samples_leaf = 1,
 min_samples_split= 4,
 n_estimators = 100,random_state=6)

rand_GSOpt.fit(x_train, y_train)
rand_GSOpt_y_pred = rand_GSOpt.predict(x_test)
print('Accuracy of Random Forest on test set: {:.2f}'.format(RandomForestModel.score(x_test, y_test)))

print('rand_GSOpt_F1 Score =',sklearn.metrics.f1_score(y_test, rand_GSOpt_y_pred, average="macro"))
print('rand_GSOpt_Precision =',sklearn.metrics.precision_score(y_test, rand_GSOpt_y_pred, average="macro"))
print('rand_GSOpt_Recall =',sklearn.metrics.recall_score(y_test, rand_GSOpt_y_pred, average="macro"))

# Ensemble Models

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import mlxtend
from mlxtend.classifier import StackingClassifier
lr=LogisticRegression()
knn=KNeighborsClassifier()
svm=SVC(probability=True)
NB=GaussianNB()
DT=DecisionTreeClassifier()

sclf=StackingClassifier(classifiers=[svm, lr, knn, DT, rand_GSOpt], use_probas=True, meta_classifier=rand_GSOpt)
sclf.fit(x_train, y_train)
sclf_y_pred = sclf.predict(x_test)
print('Accuracy of ensembeled stacking model classifier on test set: {:.2f}'.format(sclf.score(x_test, y_test)))

print('sclf_F1 Score =',sklearn.metrics.f1_score(y_test, sclf_y_pred, average="macro"))
print('sclf_Precision =',sklearn.metrics.precision_score(y_test, sclf_y_pred, average="macro"))
print('sclf_Recall =',sklearn.metrics.recall_score(y_test, sclf_y_pred, average="macro"))

import xgboost
from xgboost import XGBClassifier
XGBoost=XGBClassifier(learning_rate= 0.2, max_depth= 2, n_estimators= 43)
XGBoost.fit(x_train,y_train)
XGBoost_y_pred = XGBoost.predict(x_test)

print('Accuracy of XGBoost on test set: {:.2f}'.format(XGBoost.score(x_test, y_test)))

print('XGBoost_F1 Score =',sklearn.metrics.f1_score(y_test, XGBoost_y_pred, average="macro"))
print('XGBoost_Precision =',sklearn.metrics.precision_score(y_test, XGBoost_y_pred, average="macro"))
print('XGBoost_Recall =',sklearn.metrics.recall_score(y_test, XGBoost_y_pred, average="macro"))
# Saving the different models and std scalar file

import pickle

# Writing different model files to file
with open('Logistic_ModelForPrediction.pkl', 'wb') as f:
    pickle.dump(log_reg, f)

with open('SVM_ModelForPrediction.pkl', 'wb') as f:
    pickle.dump(classifier_rbf, f)

with open('RandomForest_ModelForPrediction.pkl', 'wb') as f:
    pickle.dump(RandomForestModel, f)

with open('RandFor_GridOpt_Model.pkl', 'wb') as f:
    pickle.dump(rand_GSOpt, f)

with open('Ensemble_Models.pkl', 'wb') as f:
    pickle.dump(sclf, f)

with open('XGBoost.pkl', 'wb') as f:
    pickle.dump(XGBoost, f)

with open('standardScalar.pkl', 'wb') as f:
    pickle.dump(scalar, f)
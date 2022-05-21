# E-Commerce Behaviour Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('online_shoppers_intention.csv')

X = train.drop(columns=["Revenue"])
X = pd.get_dummies(X,drop_first=True)
y = train["Revenue"]

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=101)

from sklearn.model_selection import GridSearchCV # Cross validation
from sklearn.ensemble import RandomForestClassifier # Random Forrest

n_estimators = [50, 64, 100, 128, 200] # Random forrest hyperparameters
max_features = [2,3,4,5,6]
param_grid = {'n_estimators':n_estimators,'max_features':max_features}

rfc = RandomForestClassifier(n_estimators=10)
grid = GridSearchCV(rfc,param_grid)
grid.fit(X_train,y_train)

predictions = grid.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
print(classification_report(y_test,predictions))



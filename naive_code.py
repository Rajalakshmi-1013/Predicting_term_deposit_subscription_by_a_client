# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 10:07:27 2021

@author: vishali jothimuthu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report
____________________________
from sklearn.datasets import load_iris
iris=load_iris()
X=iris.data
y=iris.target
# spliting X and y into training and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=1)

#_______training the model on training set
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)

y_pred=gnb.predict(X_test)
y_pred
#____comparing actual response values(y_test) with predicted response values(y_pred)
pd.crosstab(y_test,y_pred,margins=True)
#confusion matrix
cm=confusion_matrix(y_test,y_pred)
cm
#________calculating accuracy manually
(19+19+19)/(19+19+19+1+2)
from sklearn import metrics
print("Gaussian naive Bayes model accuracy(in %):",metrics.accuracy_score(y_test,y_pred)*100)

# classification report
from sklearn.metrics import classification_report 
print(classification_report(y_test,y_pred))

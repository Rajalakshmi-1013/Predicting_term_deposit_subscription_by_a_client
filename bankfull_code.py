# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 08:29:51 2021

@author: vishali jothimuthu
"""

import os
os.chdir('C:/Users/vishali jothimuthu/Desktop/Predicting Term Deposit Subscription by a client/Dataset')
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import sklearn
from sklearn import model_selection
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC 
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve,auc

df = pd.read_csv('bank-additional-full.csv',sep=";")
df=pd.DataFrame(df)
df.info()
df.shape
df.describe()
df.head(3)
df.isnull().sum()
#hist
plt.hist(df.y, bins = 'auto', facecolor = 'red')
plt.xlabel('y')
plt.ylabel('counts')
plt.title('Histogram of y')
#boxplot
import seaborn as sns
sns.countplot(df.y)
#_____________age

Q1 = np.percentile(df.age, 25, interpolation = 'midpoint')  
Q2 = np.percentile(df.age, 50, interpolation = 'midpoint')  
Q3 = np.percentile(df.age, 75, interpolation = 'midpoint')   
print('Q1 25 percentile of the given data is, ', Q1) 
print('Q1 50 percentile of the given data is, ', Q2) 
print('Q1 75 percentile of the given data is, ', Q3)   
IQR = Q3 - Q1  
print('Interquartile range is', IQR) #15

low_lim = Q1 - 1.5 * IQR 
up_lim = Q3 + 1.5 * IQR # 69.5
print('low_limit is', low_lim) #9.5
print('up_limit is', up_lim) 

df.age[df.age > 69.5] = 69.5

#_______counting outliers
len(df.age[df.age > 69.5]) # 469


sns.boxplot(df.age)

df.age.value_counts()
df.info()

#_______job 
df.job.value_counts()
sns.countplot(df.job)
'''
admin.           10422
blue-collar       9254
technician        6743
services          3969
management        2924
retired           1720
entrepreneur      1456
self-employed     1421
housemaid         1060
unemployed        1014
student            875
unknown            330
Name: job, dtype: int64
'''
# marital
df.marital.value_counts()
sns.countplot(df.marital)
___________________________________________
#selecting numerical variables

numerical_features = list(df.select_dtypes(include=['int16','int32','int64','float64']).columns)
numerical_features = list(set(numerical_features) - set(['Id']))
print("Numerical Features: ", numerical_features)

'''
'euribor3m', 'duration', 'emp.var.rate', 'previous', 'cons.conf.idx',
 'age', 'nr.employed', 'campaign', 'cons.price.idx', 'pdays'
'''
#selecting categorical vairables

categorical_features = list(df.select_dtypes(include='object').columns)
print("\nCategorical Features: ", categorical_features)

'''
'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 
'month', 'day_of_week', 'poutcome', 'y'
'''
#_____checking outliers in numerical variables

def check_outliers_continuous(df, cols):
    f, ax = plt.subplots(nrows=len(cols), figsize=(14,len(cols)+1))
    for i in range(0,len(cols)):
        sns.boxplot(df[cols[i]].dropna(), ax=ax[i])
        ax[i].title.set_text(cols[i])
    plt.tight_layout()
    plt.show()
    
check_outliers_continuous(df, numerical_features)
#_____________
def treat_outliers(df, col):
    q1, q3 = np.percentile(df[col].dropna(), [5, 95])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    #print(col, lower_bound, upper_bound)
    df[col][df[col] <= lower_bound] = lower_bound
    df[col][df[col] >= upper_bound] = upper_bound
#--------
for i in numerical_features:
    treat_outliers(df, i)

check_outliers_continuous(df, numerical_features)
--------
df.job['unemployed'] = df.job['unemployed'].astype(float)

#_________________________________

X = df.drop("y",axis=1)
y=df['y']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=0)

X.head()
y.head()
from sklearn.model_selection import train_test_split
X_train.isnull().sum()
df.info()

#_________________________________________

#Random Forest
from sklearn.ensemble import RandomForestClassifier
#Create Model with 300 trees
rfc = RandomForestClassifier(n_estimators=300)

#Fitting the model
mpr_rfc = rfc.fit(X_train, y_train)

#Prediction
y_predrfc = mpr_rfc.predict(X_test)

#Confusion Matrix & Report
pd.crosstab(y_test,y_predrfc, margins=True,rownames=['Actual'], colnames=['Predict'])


# Rows containing duplicate data
duplicate_rows_df = df[df.duplicated()]
print("number of duplicate rows:", duplicate_rows_df.shape)
#____________________________________
#duplicates
df = df.drop_duplicates()
df.head(5)
#null values
print(df.isnull().sum())
df = df.dropna() 
df.count()
#outliers
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3-Q1
print(IQR)
'''
age                15.000
duration          217.000
campaign            2.000
pdays               0.000
previous            0.000
emp.var.rate        3.200
cons.price.idx      0.919
cons.conf.idx       6.300
euribor3m           3.617
nr.employed       129.000
dtype: float64
'''
#cleared outliers

df = df[~((df < (Q1-1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]

df.shape
#correlation
plt.figure(figsize=(20,10))
c= df.corr()
sns.heatmap(c,cmap="BrBG",annot=True)
c

pip install pandas-profiling









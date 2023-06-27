#!/usr/bin/env python
# coding: utf-8

# In[190]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error,accuracy_score, confusion_matrix


# In[191]:


df = pd.read_csv(r'C:\Users\3593\Downloads\HeartAttack (1).csv',na_values='?')


# In[192]:


df.head(6)


# In[193]:


df.tail(7)


# In[194]:


df.info()


# In[195]:


df.isnull().sum()


# In[196]:


df.describe()


# In[197]:


df = df.drop(columns=["slope","ca","thal"],axis=1)


# In[198]:


df.head()


# In[199]:


df = df.dropna()


# In[200]:


df.head(5)


# In[201]:


df.info()


# In[202]:


df.isnull().sum()


# In[203]:


df["sex"].value_counts()


# In[204]:


df = pd.get_dummies(df,columns=["cp","restecg"])


# In[205]:


df.head()


# In[206]:


df.columns


# In[207]:


numerical_cols =["age","trestbps","chol","thalach","oldpeak"]
cat_cols = list(set(df.columns) - set(numerical_cols)-{"target"})


# In[208]:


df["oldpeak"].value_counts()


# In[209]:


df.columns


# In[210]:


df= df.rename(columns={"num       ":"target"})


# In[211]:


df.head()


# In[212]:


cat_cols = list(set(df.columns) - set(numerical_cols)-{"target"})
cat_cols


# In[213]:


numerical_cols


# In[214]:


df_train,df_test = train_test_split(df,test_size=0.2,random_state=42)


# In[215]:


len(df_train),len(df_test)


# In[216]:


scaler = StandardScaler()

def get_features_and_target_arrays(df,numerical_cols,cat_cols,scaler):
    x_numeric_scaled = scaler.fit_transform(df[numerical_cols])
    x_categorical = df[cat_cols].to_numpy()
    x = np.hstack((x_categorical,x_numeric_scaled))
    y = df["target"]
    
    return x,y


# In[217]:


x_train , y_train = get_features_and_target_arrays(df_train,cat_cols,numerical_cols,scaler)


# In[218]:


df_train.columns


# In[219]:


cat_cols


# In[ ]:





# In[220]:


clf = LogisticRegression()
clf.fit(x_train,y_train)


# In[221]:


x_test , y_test = get_features_and_target_arrays(df_test,cat_cols,numerical_cols,scaler)


# In[222]:


test_pred = clf.predict(x_test)


# 

# In[223]:


mean_squared_error(y_test,test_pred)


# In[224]:


accuracy_score(y_test,test_pred)


# In[225]:


confusion_matrix(y_test,test_pred)


# In[226]:


#decision tree
dc_clf = DecisionTreeClassifier()
dc_clf.fit(x_train,y_train)

dlf_pred = dc_clf.predict(x_test)
print(mean_squared_error(y_test,dlf_pred))
print(accuracy_score(y_test,dlf_pred))


# In[227]:


#Random Forest
rc_clf = DecisionTreeClassifier()
rc_clf.fit(x_train,y_train)

rc_pred = rc_clf.predict(x_test)
print(mean_squared_error(y_test,rc_pred))
print(accuracy_score(y_test,rc_pred))


# In[228]:


#SVM
svm_clf = DecisionTreeClassifier()
svm_clf.fit(x_train,y_train)

svm_pred = svm_clf.predict(x_test)
print(mean_squared_error(y_test,svm_pred))
print(accuracy_score(y_test,svm_pred))


# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as MSE
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


# In[2]:


#Import dataset
ccdefault = pd.read_csv('/Users/zilingzheng/Desktop/IE517/hw7/ccdefault.csv')


# In[3]:


#Pick feature and label
X = ccdefault[ccdefault.columns[1:-1]].values
y = ccdefault['DEFAULT'].values

# Split train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state=42)


# ## Part 1: Random forest estimators

# Fit a random forest model, try several different values for N_estimators, report in-sample accuracies.

# In[4]:


n_estimators=[1,5,25,50,100,150,200]
cvscore=[]
for i in n_estimators:
    start = time.process_time()
    forest = RandomForestClassifier(n_estimators=i)
    forest.fit(X_train, y_train)
    cv_scores = cross_val_score(forest,X_train,y_train,cv=10)
    end = time.process_time()
    times = end - start
    score = np.mean(cv_scores)
    cvscore.append(score)
    print('n_estimators = ', str(i))
    #print('cv_scores: ', cv_scores)
    print('accuracy = ', score)
    print('computation time = ', str(times), 's')
    print('')


# ## Part 2: Random forest feature importance

# Display the individual feature importance of your best model in Part 1 above using the code presented in Chapter 4 on page 136. {importances=forest.feature_importances_ }
# 

# In[5]:


forest_best = RandomForestClassifier(n_estimators=200)         
forest_best.fit(X_train, y_train)
importances = forest_best.feature_importances_
sorted_index = np.argsort(importances)[::-1]
x = range(X_train.shape[1])
labels = np.array(ccdefault.columns[1:])[sorted_index]
plt.bar(x, importances[sorted_index], tick_label=labels)
plt.xticks(rotation=90)
plt.show()


# In[6]:


print("Feature Importances:")
for i in range(X_train.shape[1]):
    print(labels[i], importances[sorted_index[i]])


# ## Part 3: Conclusions

# a) What is the relationship between n_estimators, in-sample CV accuracy and computation time?  
# As n_estimators increases, in-sample CV accuracy increases, also computation time increases. Therefore, when we have more estimators numbers, we get higher accuracy with longer time.
# 
# 
# b) What is the optimal number of estimators for your forest?   
# The optimal number of estimators for my forest is 200.
# 
# 
# c) Which features contribute the most importance in your model according to scikit-learn function?  
# According to scikit-learn function, the feature PAY_0 contributes the most importance in my model. 
# 
# d) What is feature importance and how is it calculated?  (If you are not sure, refer to the Scikit-Learn.org documentation.)  
# Feature importance is an array that contain positive floats indicating relative importance of the corresponding features. It is calculated as the averageed impurity decrease from all decision trees in the forest, and the higher value the more important the feature.
# 

# ## Part 4: Appendix

# Link to GitHub: https://github.com/ZilingZheng/IE517_F20/blob/master/IE517_FA20_HW7.py

# In[7]:


print("My name is Ziling Zheng")
print("My NetID is: zzheng27")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


# In[2]:


#Import dataset
ccdefault = pd.read_csv('/Users/zilingzheng/Desktop/IE517/hw6/ccdefault.csv')
ccdefault.head()


# In[3]:


#Pick feature and label
X = ccdefault[ccdefault.columns[1:-1]].values
y = ccdefault['DEFAULT'].values


# ## Part 1: Random test train splits

# Run in-sample and out-of-sample accuracy scores for 10 different samples by changing random_state from 1 to 10 in sequence. 
# Display the individual scores, then calculate the mean and standard deviation on the set of scores.  Report in a table format.

# In[4]:


score_train=[]
score_test=[]
start_rs= time.process_time()
for i in range(1,11):
    # Split train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, stratify=y, random_state=i)
    tree = DecisionTreeClassifier(random_state=i)
    tree.fit(X_train,y_train)
    y_train_pred = tree.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_train_pred)
    y_test_pred = tree.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    score_train.append(accuracy_train)
    score_test.append(accuracy_test)

end_rs = time.process_time()
print("train accuracy scores: {}".format(score_train))
print("test accuracy scores: {}".format(score_test))
print("running time for random state: {}s".format(end_rs-start_rs))


# In[5]:


rs=np.arange(1,11)
plt.plot(rs, score_train)
plt.plot(rs, score_test)
plt.xlabel('ramdom_state')
plt.ylabel('Accuracy')
plt.legend(['train', 'test'])
plt.show()


# In[6]:


print("train mean of accuracy: {}".format(np.mean(score_train)))
print("test mean of accuracy: {}".format(np.mean(score_test)))
print("train standard deviation of accuracy: {}".format(np.std(score_train)))
print("test standard deviation of accuracy: {}".format(np.std(score_test)))


# |*random state*|train|test|
# |-|-|-|
# |**mean**|0.9994|0.7253|
# |**standard deviation**|0.0000474|0.0081041|

# ## Part 2: Cross validation

# Now rerun your model using cross_val_scores with k-fold CV (k=10).  
# Report the individual fold accuracy scores, the mean CV score and the standard deviation of the fold scores.  Now run the out-of-sample accuracy score.  Report in a table format.

# In[14]:


start_cv= time.process_time()
tree = DecisionTreeClassifier()
cv_scores_test = cross_val_score(tree, X, y, cv=10,n_jobs=-1)
end_cv= time.process_time()
print("the individual fold accuracy scores: {}".format(cv_scores))
print("the mean CV scores: {}".format(cv_scores.mean()))
print("the standard deviation of the fold scores: {}".format(cv_scores.std()))
print("running time for cv: {}s".format(end_cv-start_cv))


# |*k=10*|test|
# |-|-|
# |**mean**|0.7269|
# |**standard deviation**|0.0091|

# In[8]:


k=np.arange(1,11)
plt.plot(rs, cv_scores)
plt.xlabel('fold')
plt.ylabel('Accuracy')
plt.show()


# ## Part 3: Conclusions

# Write a short paragraph summarizing your findings.  Which method of measuring accuracy provides the best estimate of how a model will do against unseen data?  Which one is more efficient to run?

# The accuracy from both random test train splits and cross validation methods are similar, around 0.73. However, by looking at the running time, cross validation method is much faster than random test train splits method. Therefore, cross validation method is more efficient to run. 

# ## Appendix

# Link to GitHub: https://github.com/ZilingZheng/IE517_F20/blob/master/IE517_FA20_HW6.py

# In[16]:


print("My name is Ziling Zheng")
print("My NetID is: zzheng27")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


# In[ ]:





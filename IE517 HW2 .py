#!/usr/bin/env python
# coding: utf-8

# In[164]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn import datasets
from sklearn import preprocessing
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE


# In[165]:


#Import dataset
df = pd.read_csv('/Users/zilingzheng/Desktop/IE517/hw2/Treasury Squeeze test - DS1.csv')

#Pick feature and label
X,y=df.iloc[:,2:11],df.iloc[:,11]
y=y.map({True:1,False:0})
#Split train and test sets 
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#Standardize the features
scaler= preprocessing.StandardScaler().fit(X_train)
X_train= scaler.transform(X_train)
X_test= scaler.transform(X_test)


# In[166]:


#KNN
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
print(knn.score(X_train,y_train))
print(knn.score(X_test, y_test))

#Find the best k
k_range= range(1,26)
scores=[]
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

print(scores)
plt.plot(scores)

print('From the graph, we can find that the best k=12.')


# In[167]:


#Decision Tree

def plot_decision_regions(X, y, classifier, test_idx=None,
                          resolution=0.02):
    #setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    #plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y ==cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl,
                    edgecolor='black')

    #highlight test samples
    if test_idx:
        #plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o',
                    s=100, label='test set')


# In[168]:



def gini(p):
    return (p)*(1-(p)) + (1-p)*(1-(1-p))
def entropy(p):
    return - p*np.log2(p) - (1-p)*np.log2((1-p))
def error(p):
    return 1 - np.max([p, 1-p])
x = np.arange(0.0, 1.0, 0.01)
ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e*0.5 if e else None for e in ent]
err = [error(i) for i in x]
fig = plt.figure()
ax = plt.subplot(111)
for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err],
                         ['Entropy','Entropy(scaled)','Gini Impurity',
                              'Misclassification Error'],
                         ['-','-','--','-.'],
                         ['black','lightgray',
                              'red','green','cyan']):
    line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
         ncol=5, fancybox=True, shadow=False)
ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')
plt.show()


# In[169]:


#Import dataset
df = pd.read_csv('/Users/zilingzheng/Desktop/IE517/hw2/Treasury Squeeze test - DS1.csv')
#df.head()

#Pick feature and label
X,y=df.iloc[:,2:4],df.iloc[:,11]
y=y.map({True:1,False:0})
#Split train and test sets 
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
#Standardize the features
scaler= preprocessing.StandardScaler().fit(X_train)
X_train= scaler.transform(X_train)
X_test= scaler.transform(X_test)

tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
tree.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined, y_combined, classifier=tree, test_idx=range(631,900))
plt.xlabel('feature1')
plt.ylabel('feature2')
plt.legend(loc='upper left')
plt.show()


# In[170]:



print("My name is Ziling Zheng")
print("My NetID is: zzheng27")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


# In[ ]:





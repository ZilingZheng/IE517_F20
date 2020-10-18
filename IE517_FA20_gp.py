#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import pylab
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.svm import SVC
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier as KNN


# # Chapter 1:  CreditScore

# #### 1) Introduction/Exploratory Data Analysis

# In[2]:


#Import dataset
creditscore_original = pd.read_csv('/Users/zilingzheng/Desktop/IE517/gp/MLF_GP1_CreditScore.csv')
creditscore_original.head()


# In[3]:


#print dimension of data frame
nrow = creditscore_original.shape[0]
ncol = creditscore_original.shape[1]
print("Number of Rows of Data = ", nrow) 
print("Number of Columns of Data = ", ncol)


# In[4]:


#print summary of data frame
creditscore_original.describe()


# In[5]:


creditscore_original.info()


# In[6]:


print('Number of "Investment Grade": ', creditscore_original['InvGrd'].value_counts()[1])
print('Number of "Not Investment Grade": ', creditscore_original['InvGrd'].value_counts()[0])


# In[7]:


creditscore_original['Rating'].value_counts()


# In[8]:


class_mapping = {'A1':1, 'A2':2, 'A3':3, 'Aa2':4, 'Aa3':5, 'Aaa':6, 
                 'B1':7, 'B2':8, 'B3':9, 'Ba1':10, 'Ba2':11, 'Ba3':12,
                 'Baa1':13, 'Baa2':14, 'Baa3':15, 'Caa1':16}
rating_num = creditscore_original['Rating'].map(class_mapping)
creditscore = creditscore_original.iloc[:,:-1]
creditscore['rating_num'] = rating_num
creditscore.head()


# In[9]:


creditscore.info()


# In[10]:


#ECDF
def ecdf(data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1) / n
    return x, y

x_Rating, y_Rating = ecdf(creditscore_original['Rating'])
plt.figure(figsize=(12,5))
plt.plot(x_Rating, y_Rating, marker = '.', linestyle ='-')
plt.xlabel('Moodys credit rating')
plt.ylabel('ECDF')
plt.title('Moodys credit rating VS ECDF')
plt.show()


# In[11]:


#Scatterplot Matrix
sns.set(style='whitegrid', context='notebook')
sns.pairplot(creditscore, height=2.5)
plt.show()


# In[12]:


sns.set(style='whitegrid', context='notebook')
cols = ['Sales/Revenues', 'EBITDA', 'Free Cash Flow', 'Current Liabilities',
        'EPS Before Extras', 'PE', 'ROA', 'ROE', 'InvGrd', 'rating_num']
sns.pairplot(creditscore[cols], height=2.5)
plt.show()


# In[13]:


corMat = pd.DataFrame(creditscore.corr())
corMat


# In[14]:


cm = corMat
plt.figure(figsize=(25,25))
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15})


plt.title('Heatmap for Credit Score', fontsize=36, fontweight='bold')

plt.show()


# #### 2) Preprocessing, feature extraction, feature selection

# In[15]:


#Pick feature and label
X = creditscore[creditscore.columns[:-2]].values
y_inv = creditscore['InvGrd'].values
y_rate = creditscore['rating_num'].values

# Split train and test sets
X_train_inv, X_test_inv, y_train_inv, y_test_inv = train_test_split(X, y_inv, test_size = 0.1, random_state=42)
X_train_rate, X_test_rate, y_train_rate, y_test_rate = train_test_split(X, y_rate, test_size = 0.1, random_state=42)

#Standardize the features
scaler= StandardScaler()
X_train_std_inv= scaler.fit_transform(X_train_inv)
X_test_std_inv= scaler.transform(X_test_inv)
X_train_std_rate= scaler.fit_transform(X_train_rate)
X_test_std_rate= scaler.transform(X_test_rate)


# In[16]:


# Create a PCA model for all components: pca
pca= PCA()

#Transform 
X_train_inv_pca = pca.fit_transform(X_train_std_inv)
X_test_inv_pca = pca.transform(X_test_std_inv)
X_train_rate_pca = pca.fit_transform(X_train_std_rate)
X_test_rate_pca = pca.transform(X_test_std_rate)


# In[17]:


# Plot the explained variances (credit score)
features = range(pca.n_components_)
plt.figure(figsize=(12,5))
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.title('PCA explained variances for credit score', fontsize=16, fontweight='bold')
plt.show()


# In[18]:


#pick n_components=3
pca_inv = PCA(n_components=3)
pca_rate = PCA(n_components=3)

#Transform 
X_train_inv_pca = pca_inv.fit_transform(X_train_std_inv)
X_test_inv_pca = pca_inv.transform(X_test_std_inv)
X_train_rate_pca = pca_rate.fit_transform(X_train_std_rate)
X_test_rate_pca = pca_rate.transform(X_test_std_rate)


# #### 3) Model fitting and evaluation 

# **Target: Investment Grade**

# In[19]:


#Logistic Regression 
#without PCA
print("Logistic Regression without PCA")
print("-------------------------------------")
X_train_inv_without= scaler.transform(X_train_inv)
X_test_inv_without= scaler.transform(X_test_inv)
logreg= LogisticRegression(solver='lbfgs')
logreg.fit(X_train_inv_without, y_train_inv)

# Compute R^2 
R2_inv = logreg.score(X_test_inv_without, y_test_inv)
print("R^2 test: {}".format(R2_inv))
#Compute Mean Squared Error
y_inv_pred = logreg.predict(X_test_inv_without)
mse_inv = mean_squared_error(y_test_inv, y_inv_pred)
print("MSE test: {}".format(mse_inv))
print("-------------------------------------")


# In[20]:


#with PCA
print("Logistic Regression with PCA")
print("-------------------------------------")
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train_inv_pca, y_train_inv)

# Compute R^2 
R2_inv = logreg.score(X_test_inv_pca, y_test_inv)
print("R^2 test: {}".format(R2_inv))
#Compute Mean Squared Error
y_inv_pred = logreg.predict(X_test_inv_pca)
mse_inv = mean_squared_error(y_test_inv, y_inv_pred)
print("MSE test: {}".format(mse_inv))
print("-------------------------------------")


# In[21]:


#KNN
#without pca
print("KNN without PCA")
print("-------------------------------------")
knn=KNN()
knn.fit(X_train_inv_without, y_train_inv)

# Compute R^2 
R2_inv = knn.score(X_test_inv_without, y_test_inv)
print("R^2 test: {}".format(R2_inv))
#Compute Mean Squared Error
y_inv_pred = knn.predict(X_test_inv_without)
mse_inv = mean_squared_error(y_test_inv, y_inv_pred)
print("MSE test: {}".format(mse_inv))
print("-------------------------------------")


# In[22]:


#with pca
print("KNN with PCA")
print("-------------------------------------")
knn=KNN()
knn.fit(X_train_inv_pca, y_train_inv)

# Compute R^2 
R2_inv = knn.score(X_test_inv_pca, y_test_inv)
print("R^2 test: {}".format(R2_inv))
#Compute Mean Squared Error
y_inv_pred = knn.predict(X_test_inv_pca)
mse_inv = mean_squared_error(y_test_inv, y_inv_pred)
print("MSE test: {}".format(mse_inv))
print("-------------------------------------")


# In[23]:


#Decision Tree Classifier
#without pca
print("Decision Tree Classifier without PCA")
print("-------------------------------------")
tree=DecisionTreeClassifier()
tree.fit(X_train_inv_without, y_train_inv)

# Compute R^2 
R2_inv = tree.score(X_test_inv_without, y_test_inv)
print("R^2 test: {}".format(R2_inv))
#Compute Mean Squared Error
y_inv_pred = tree.predict(X_test_inv_without)
mse_inv = mean_squared_error(y_test_inv, y_inv_pred)
print("MSE test: {}".format(mse_inv))
print("-------------------------------------")


# In[24]:


#with pca
print("Decision Tree Classifier with PCA")
print("-------------------------------------")
tree=DecisionTreeClassifier()
tree.fit(X_train_inv_pca, y_train_inv)

# Compute R^2 
R2_inv = tree.score(X_test_inv_pca, y_test_inv)
print("R^2 test: {}".format(R2_inv))
#Compute Mean Squared Error
y_inv_pred = tree.predict(X_test_inv_pca)
mse_inv = mean_squared_error(y_test_inv, y_inv_pred)
print("MSE test: {}".format(mse_inv))
print("-------------------------------------")


# In[25]:


#SVC
#without pca
print("SVC without PCA")
print("-------------------------------------")
svc=SVC()
svc.fit(X_train_inv_without, y_train_inv)

# Compute R^2 
R2_inv = svc.score(X_test_inv_without, y_test_inv)
print("R^2 test: {}".format(R2_inv))
#Compute Mean Squared Error
y_inv_pred = svc.predict(X_test_inv_without)
mse_inv = mean_squared_error(y_test_inv, y_inv_pred)
print("MSE test: {}".format(mse_inv))
print("-------------------------------------")


# In[26]:


#with pca
print("SVC with PCA")
print("-------------------------------------")
svc=SVC(gamma='scale')
svc.fit(X_train_inv_pca, y_train_inv)

# Compute R^2 
R2_inv = svc.score(X_test_inv_pca, y_test_inv)
print("R^2 test: {}".format(R2_inv))
#Compute Mean Squared Error
y_inv_pred = svc.predict(X_test_inv_pca)
mse_inv = mean_squared_error(y_test_inv, y_inv_pred)
print("MSE test: {}".format(mse_inv))
print("-------------------------------------")


# **Target: Moody's credit rating'**

# In[27]:


#Logistic Regression 
#without PCA
print("Logistic Regression without PCA")
print("-------------------------------------")
X_train_rate_without= scaler.transform(X_train_rate)
X_test_rate_without= scaler.transform(X_test_rate)
logreg= LogisticRegression(solver='lbfgs', multi_class = 'auto', max_iter=1000)
logreg.fit(X_train_rate_without, y_train_rate)

# Compute R^2 
R2_rate = logreg.score(X_test_rate_without, y_test_rate)
print("R^2 test: {}".format(R2_rate))
#Compute Mean Squared Error
y_rate_pred = logreg.predict(X_test_rate_without)
mse_rate = mean_squared_error(y_test_rate, y_rate_pred)
print("MSE test: {}".format(mse_rate))
print("-------------------------------------")


# In[28]:


#with PCA
print("Logistic Regression with PCA")
print("-------------------------------------")
logreg = LogisticRegression(solver='lbfgs', multi_class = 'auto', max_iter=1000)
logreg.fit(X_train_rate_pca, y_train_rate)

# Compute R^2 
R2_rate = logreg.score(X_test_rate_pca, y_test_rate)
print("R^2 test: {}".format(R2_rate))
#Compute Mean Squared Error
y_rate_pred = logreg.predict(X_test_rate_pca)
mse_rate = mean_squared_error(y_test_rate, y_rate_pred)
print("MSE test: {}".format(mse_rate))
print("-------------------------------------")


# In[29]:


#KNN
#without pca
print("KNN without PCA")
print("-------------------------------------")
knn=KNN()
knn.fit(X_train_rate_without, y_train_rate)

# Compute R^2 
R2_rate = knn.score(X_test_rate_without, y_test_rate)
print("R^2 test: {}".format(R2_rate))
#Compute Mean Squared Error
y_rate_pred = knn.predict(X_test_rate_without)
mse_rate = mean_squared_error(y_test_rate, y_rate_pred)
print("MSE test: {}".format(mse_rate))
print("-------------------------------------")


# In[30]:


#with pca
print("KNN with PCA")
print("-------------------------------------")
knn=KNN()
knn.fit(X_train_rate_pca, y_train_rate)

# Compute R^2 
R2_rate = knn.score(X_test_rate_pca, y_test_rate)
print("R^2 test: {}".format(R2_rate))
#Compute Mean Squared Error
y_rate_pred = knn.predict(X_test_rate_pca)
mse_rate = mean_squared_error(y_test_rate, y_rate_pred)
print("MSE test: {}".format(mse_rate))
print("-------------------------------------")


# In[31]:


#Decision Tree Classifier
#without pca
print("Decision Tree Classifier without PCA")
print("-------------------------------------")
tree=DecisionTreeClassifier()
tree.fit(X_train_rate_without, y_train_rate)

# Compute R^2 
R2_rate = tree.score(X_test_rate_without, y_test_rate)
print("R^2 test: {}".format(R2_rate))
#Compute Mean Squared Error
y_rate_pred = tree.predict(X_test_rate_without)
mse_rate = mean_squared_error(y_test_rate, y_rate_pred)
print("MSE test: {}".format(mse_rate))
print("-------------------------------------")


# In[32]:


#with pca
print("Decision Tree Classifier with PCA")
print("-------------------------------------")
tree=DecisionTreeClassifier()
tree.fit(X_train_rate_pca, y_train_rate)

# Compute R^2 
R2_rate = tree.score(X_test_rate_pca, y_test_rate)
print("R^2 test: {}".format(R2_rate))
#Compute Mean Squared Error
y_rate_pred = tree.predict(X_test_rate_pca)
mse_rate = mean_squared_error(y_test_rate, y_rate_pred)
print("MSE test: {}".format(mse_rate))
print("-------------------------------------")


# In[33]:


#SVC
#without pca
print("SVC without PCA")
print("-------------------------------------")
svc=SVC()
svc.fit(X_train_rate_without, y_train_rate)

# Compute R^2 
R2_rate = svc.score(X_test_rate_without, y_test_rate)
print("R^2 test: {}".format(R2_rate))
#Compute Mean Squared Error
y_rate_pred = svc.predict(X_test_rate_without)
mse_rate = mean_squared_error(y_test_rate, y_rate_pred)
print("MSE test: {}".format(mse_rate))
print("-------------------------------------")


# In[34]:


#with pca
print("SVC with PCA")
print("-------------------------------------")
svc=SVC(gamma='scale')
svc.fit(X_train_rate_pca, y_train_rate)

# Compute R^2 
R2_rate = svc.score(X_test_rate_pca, y_test_rate)
print("R^2 test: {}".format(R2_rate))
#Compute Mean Squared Error
y_rate_pred = svc.predict(X_test_rate_pca)
mse_rate = mean_squared_error(y_test_rate, y_rate_pred)
print("MSE test: {}".format(mse_rate))
print("-------------------------------------")


# #### 4) Hyperparameter tuning

# For target Investment Grade, since KNN performs the best, so we are going to optimize its hyperparameters.

# In[35]:


k_space = np.arange(1,30)
param_grid = {'n_neighbors': k_space}
knn=KNN()
knn_cv= GridSearchCV(knn,param_grid,cv=10)
knn_cv.fit(X_train_inv_pca, y_train_inv)
print("Tuned KNN Parameter: {}".format(knn_cv.best_params_))
print("Tuned KNN Accuracy: {}".format(knn_cv.best_score_))
k=int(knn_cv.best_params_['n_neighbors'])


# For target Moody's credit rating, since Decision Tree Classifier performs the best, so we are going to optimize its hyperparameters.

# In[36]:


warnings.filterwarnings("ignore")
max_depth = np.arange(1,30)
param_grid = {"criterion": ["gini", "entropy"],
              'max_depth': max_depth}
tree = DecisionTreeClassifier()
#tree_cv = RandomizedSearchCV(tree,param_grid,cv=10)
tree_cv = GridSearchCV(tree,param_grid,cv=10)
tree_cv.fit(X_train_rate_without, y_train_rate)
print("Tuned Decision Tree Classifier Parameter: {}".format(tree_cv.best_params_))
print("Tuned Decision Tree Classifier Accuracy: {}".format(tree_cv.best_score_))


# #### 5) Ensembling 

# **Target: Investment Grade**

# Bagging

# In[37]:


knn = KNN(n_neighbors=k)
bc = BaggingClassifier(base_estimator=knn, 
            n_estimators=200,
            oob_score=True,
            random_state=1)

bc.fit(X_train_inv_without, y_train_inv)

y_pred_inv = bc.predict(X_test_inv_without)
acc_test = accuracy_score(y_test_inv, y_inv_pred)

acc_oob = bc.oob_score_

print('Test set accuracy: {:.3f}, OOB accuracy: {:.3f}'.format(acc_test, acc_oob))


# Adaboost

# In[38]:


dt = DecisionTreeClassifier(max_depth=20)
ada = AdaBoostClassifier(base_estimator=dt, n_estimators=200)
ada.fit(X_train_inv_without, y_train_inv)
y_pred_proba = ada.predict_proba(X_test_inv_without)[:,1]

from sklearn.metrics import roc_auc_score
ada_roc_auc = roc_auc_score(y_test_inv, y_pred_proba)

# Print roc_auc_score
print('ROC AUC score: {:.2f}'.format(ada_roc_auc))


# Random Forest 

# In[39]:


forest = RandomForestClassifier(n_estimators=200, max_depth=20)         
forest.fit(X_train_inv_without, y_train_inv)
importances = forest.feature_importances_
sorted_index = np.argsort(importances)[::-1]
x = range(X_train_inv_without.shape[1])
labels = np.array(creditscore.columns[:-2])[sorted_index]
plt.figure(figsize=(15,6))
plt.bar(x, importances[sorted_index], tick_label=labels)
plt.title("Feature Importances for Investment Grade", fontsize=16, fontweight='bold')
plt.xticks(rotation=90)
plt.show()


# In[40]:


cv_scores = cross_val_score(forest,X_train_inv_without, y_train_inv,cv=10)
trainscore = np.mean(cv_scores)
print('in sample accuracy = ', trainscore)

y_inv_pred = forest.predict(X_test_inv_without)
testmse = mean_squared_error(y_test_inv, y_inv_pred)
testscore = accuracy_score(y_test_inv, y_inv_pred)
print('out sample accuracy = ', testscore)
print('out sample MSE = ', testmse)


# In[ ]:





# **Target: Moody's credit rating**

# Random Forest

# In[41]:


forest = RandomForestClassifier(n_estimators=200, max_depth=20)         
forest.fit(X_train_rate_without, y_train_rate)
importances = forest.feature_importances_
sorted_index = np.argsort(importances)[::-1]
x = range(X_train_rate_without.shape[1])
labels = np.array(creditscore.columns[:-2])[sorted_index]
plt.figure(figsize=(15,6))
plt.bar(x, importances[sorted_index], tick_label=labels)
plt.xticks(rotation=90)
plt.title("Feature Importances for Moody's credit rating", fontsize=16, fontweight='bold')
plt.show()


# In[42]:


cv_scores = cross_val_score(forest,X_train_rate_without, y_train_rate, cv=10)

trainscore = np.mean(cv_scores)
print('in sample accuracy = ', trainscore)

y_rate_pred = forest.predict(X_test_rate_without)
testmse = mean_squared_error(y_test_rate, y_rate_pred)
testscore = accuracy_score(y_test_rate, y_rate_pred)
print('out sample accuracy = ', testscore)
print('out sample MSE = ', testmse)


# # Chapter 2: EconCycle

# #### 1) Introduction/Exploratory Data Analysis

# In[127]:


#Import dataset
econcycle_all = pd.read_csv('/Users/zilingzheng/Desktop/IE517/gp/MLF_GP2_EconCycle.csv')
econcycle_all.head()


# In[128]:


econcycle=econcycle_all.drop(["Date"],axis =1)
econcycle.head()


# In[129]:


econcycle.info()


# In[130]:


#print dimension of data frame
nrow = econcycle.shape[0]
ncol = econcycle.shape[1]
print("Number of Rows of Data = ", nrow) 
print("Number of Columns of Data = ", ncol)


# In[131]:


#print summary of data frame
econcycle.describe()


# In[132]:


plt.figure(figsize=(20,5))
plt.plot(econcycle_all['Date'],econcycle['PCT 3MO FWD'], marker = '', linestyle ='-')
plt.plot(econcycle_all['Date'],econcycle['PCT 6MO FWD'], marker = '', linestyle ='-')
plt.plot(econcycle_all['Date'],econcycle['PCT 9MO FWD'], marker = '', linestyle ='-')
plt.legend(('PCT 3MO FWD', 'PCT 6MO FWD', 'PCT 9MO FWD'))
x_ticks = ['1/31/1979', '7/31/1979', '1/31/1980', '7/31/1980', '1/31/1981', '7/31/1981', '1/31/1982', '7/31/1982',
           '1/31/1983', '7/31/1983', '1/31/1984', '7/31/1984', '1/31/1985', '7/31/1985', '1/31/1986', '7/31/1986',
           '1/31/1987', '7/31/1987', '1/31/1988', '7/31/1988', '1/31/1989', '7/31/1989', '1/31/1990', '7/31/1990',
           '1/31/1991', '7/31/1991', '1/31/1992', '7/31/1992', '1/31/1993', '7/31/1993', '1/31/1994', '7/31/1994',
           '1/31/1995', '7/31/1995', '1/31/1996', '7/31/1996','1/31/1997','7/31/1997']
x_ticks_label = ['1/31/1979', '7/31/1979', '1/31/1980', '7/31/1980', '1/31/1981', '7/31/1981', '1/31/1982', '7/31/1982',
                 '1/31/1983', '7/31/1983', '1/31/1984', '7/31/1984', '1/31/1985', '7/31/1985', '1/31/1986', '7/31/1986',
                 '1/31/1987', '7/31/1987', '1/31/1988', '7/31/1988', '1/31/1989', '7/31/1989', '1/31/1990', '7/31/1990',
                 '1/31/1991', '7/31/1991', '1/31/1992', '7/31/1992', '1/31/1993', '7/31/1993', '1/31/1994', '7/31/1994',
                 '1/31/1995', '7/31/1995', '1/31/1996', '7/31/1996','1/31/1997','7/31/1997']
plt.xticks(x_ticks,x_ticks_label,rotation=45)
plt.title('percent change in the USHPCI 3-6-9 months ahead', fontsize=20, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('PCT')
plt.show()


# In[133]:


plt.figure(figsize=(20,5))
plt.plot(econcycle_all['Date'],econcycle['USPHCI'], marker = '', linestyle ='-')
x_ticks = ['1/31/1979', '7/31/1979', '1/31/1980', '7/31/1980', '1/31/1981', '7/31/1981', '1/31/1982', '7/31/1982',
           '1/31/1983', '7/31/1983', '1/31/1984', '7/31/1984', '1/31/1985', '7/31/1985', '1/31/1986', '7/31/1986',
           '1/31/1987', '7/31/1987', '1/31/1988', '7/31/1988', '1/31/1989', '7/31/1989', '1/31/1990', '7/31/1990',
           '1/31/1991', '7/31/1991', '1/31/1992', '7/31/1992', '1/31/1993', '7/31/1993', '1/31/1994', '7/31/1994',
           '1/31/1995', '7/31/1995', '1/31/1996', '7/31/1996','1/31/1997','7/31/1997']
x_ticks_label = ['1/31/1979', '7/31/1979', '1/31/1980', '7/31/1980', '1/31/1981', '7/31/1981', '1/31/1982', '7/31/1982',
                 '1/31/1983', '7/31/1983', '1/31/1984', '7/31/1984', '1/31/1985', '7/31/1985', '1/31/1986', '7/31/1986',
                 '1/31/1987', '7/31/1987', '1/31/1988', '7/31/1988', '1/31/1989', '7/31/1989', '1/31/1990', '7/31/1990',
                 '1/31/1991', '7/31/1991', '1/31/1992', '7/31/1992', '1/31/1993', '7/31/1993', '1/31/1994', '7/31/1994',
                 '1/31/1995', '7/31/1995', '1/31/1996', '7/31/1996','1/31/1997','7/31/1997']
plt.xticks(x_ticks,x_ticks_label,rotation=45)
plt.xlabel('Date')
plt.ylabel('USPHCI')
plt.title('Index of USPHCI ', fontsize=20, fontweight='bold')
plt.show()


# In[134]:


#Scatterplot Matrix
sns.set(style='whitegrid', context='notebook')
sns.pairplot(econcycle, height=2.5)
plt.show()


# In[135]:


corMat = pd.DataFrame(econcycle.corr())
corMat


# In[136]:


cm = corMat
plt.figure(figsize=(20,20))
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15})
plt.title('Heatmap for Econcycle', fontsize=36, fontweight='bold')
plt.show()


# #### 2) Preprocessing, feature extraction, feature selection

# In[137]:


#Pick feature and label
X_ec = econcycle[econcycle.columns[:-3]].values
y_ec = econcycle[econcycle.columns[-3:]].values
# Split train and test sets
X_train_ec, X_test_ec, y_train_ec, y_test_ec = train_test_split(X_ec, y_ec, test_size = 0.1, random_state=42)
#Standardize the features
scaler = StandardScaler()
X_train_std_ec = scaler.fit_transform(X_train_ec)
X_test_std_ec = scaler.transform(X_test_ec)


# In[138]:


y3_train_ec = y_train_ec[:,0]
y6_train_ec = y_train_ec[:,1]
y9_train_ec = y_train_ec[:,2]
y3_test_ec = y_test_ec[:,0]
y6_test_ec = y_test_ec[:,1]
y9_test_ec = y_test_ec[:,2]


# In[139]:


# Create a PCA model for all components: pca
pca_ec = PCA()

#Transform 
X_train_ec_pca = pca_ec.fit_transform(X_train_std_ec)
X_test_ec_pca = pca_ec.transform(X_test_std_ec)


# In[140]:


# Plot the explained variances (investment grade)
features_ec = range(pca_ec.n_components_)
plt.figure(figsize=(12,5))
plt.bar(features_ec, pca_ec.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features_ec)
plt.title('PCA explained variances for econcycle', fontsize=16, fontweight='bold')
plt.show()


# In[141]:


#pick n_components=2
pca_ec = PCA(n_components=2)

#Transform 
X_train_ec_pca = pca_ec.fit_transform(X_train_std_ec)
X_test_ec_pca = pca_ec.transform(X_test_std_ec)


# #### 3) Model fitting and evaluation 

# percent change in the USHPCI 3 months ahead

# In[142]:


#Linear Regression 
print("Linear Regression")
print("-------------------------------------")
lr = LinearRegression()
lr.fit(X_train_ec_pca, y3_train_ec)

# Compute R^2 
R2_ec = lr.score(X_test_ec_pca, y3_test_ec)
print("R^2 test: {}".format(R2_ec))
#Compute Mean Squared Error
y_ec_pred = lr.predict(X_test_ec_pca)
mse_ec = mean_squared_error(y3_test_ec, y_ec_pred)
print("MSE test: {}".format(mse_ec))
print("-------------------------------------")


# In[143]:


#SVM 
print("SVM")
print("-------------------------------------")
svm = SVR()
svm.fit(X_train_ec_pca, y3_train_ec)

# Compute R^2 
R2_ec = svm.score(X_test_ec_pca, y3_test_ec)
print("R^2 test: {}".format(R2_ec))
#Compute Mean Squared Error
y_ec_pred = svm.predict(X_test_ec_pca)
mse_ec = mean_squared_error(y3_test_ec, y_ec_pred)
print("MSE test: {}".format(mse_ec))
print("-------------------------------------")


# In[144]:


#Lasso Regression 
print("Lasso Regression")
print("-------------------------------------")
lasso = Lasso()
lasso.fit(X_train_ec_pca, y3_train_ec)

# Compute R^2 
R2_ec = lasso.score(X_test_ec_pca, y3_test_ec)
print("R^2 test: {}".format(R2_ec))
#Compute Mean Squared Error
y_ec_pred = lasso.predict(X_test_ec_pca)
mse_ec = mean_squared_error(y3_test_ec, y_ec_pred)
print("MSE test: {}".format(mse_ec))
print("-------------------------------------")


# In[145]:


#Ridge Regression 
print("Ridge Regression")
print("-------------------------------------")
ridge = Ridge(normalize= True)
ridge.fit(X_train_ec_pca, y3_train_ec)

# Compute R^2 
R2_ec = ridge.score(X_test_ec_pca, y3_test_ec)
print("R^2 test: {}".format(R2_ec))
#Compute Mean Squared Error
y_ec_pred = ridge.predict(X_test_ec_pca)
mse_ec = mean_squared_error(y3_test_ec, y_ec_pred)
print("MSE test: {}".format(mse_ec))
print("-------------------------------------")


# In[146]:


#Decision Tree Regressor
print("Decision Tree Regressor")
print("-------------------------------------")
tree=DecisionTreeRegressor()
tree.fit(X_train_ec_pca, y3_train_ec)
# Compute R^2 
R2_ec = tree.score(X_test_ec_pca, y3_test_ec)
print("R^2 test: {}".format(R2_ec))
#Compute Mean Squared Error
y_ec_pred = tree.predict(X_test_ec_pca)
mse_ec = mean_squared_error(y3_test_ec, y_ec_pred)
print("MSE test: {}".format(mse_ec))
print("-------------------------------------")


# percent change in the USHPCI 6 months ahead

# In[147]:


#Linear Regression 
print("Linear Regression")
print("-------------------------------------")
lr = LinearRegression()
lr.fit(X_train_ec_pca, y6_train_ec)

# Compute R^2 
R2_ec = lr.score(X_test_ec_pca, y6_test_ec)
print("R^2 test: {}".format(R2_ec))
#Compute Mean Squared Error
y_ec_pred = lr.predict(X_test_ec_pca)
mse_ec = mean_squared_error(y6_test_ec, y_ec_pred)
print("MSE test: {}".format(mse_ec))
print("-------------------------------------")


# In[148]:


#SVM 
print("SVM")
print("-------------------------------------")
svm = SVR()
svm.fit(X_train_ec_pca, y6_train_ec)

# Compute R^2 
R2_ec = svm.score(X_test_ec_pca, y6_test_ec)
print("R^2 test: {}".format(R2_ec))
#Compute Mean Squared Error
y_ec_pred = svm.predict(X_test_ec_pca)
mse_ec = mean_squared_error(y6_test_ec, y_ec_pred)
print("MSE test: {}".format(mse_ec))
print("-------------------------------------")


# In[149]:


#Lasso Regression 
print("Lasso Regression")
print("-------------------------------------")
lasso = Lasso()
lasso.fit(X_train_ec_pca, y6_train_ec)

# Compute R^2 
R2_ec = lasso.score(X_test_ec_pca, y6_test_ec)
print("R^2 test: {}".format(R2_ec))
#Compute Mean Squared Error
y_ec_pred = lasso.predict(X_test_ec_pca)
mse_ec = mean_squared_error(y6_test_ec, y_ec_pred)
print("MSE test: {}".format(mse_ec))
print("-------------------------------------")


# In[150]:


#Ridge Regression 
print("Ridge Regression")
print("-------------------------------------")
ridge = Ridge(normalize= True)
ridge.fit(X_train_ec_pca, y6_train_ec)

# Compute R^2 
R2_ec = ridge.score(X_test_ec_pca, y6_test_ec)
print("R^2 test: {}".format(R2_ec))
#Compute Mean Squared Error
y_ec_pred = ridge.predict(X_test_ec_pca)
mse_ec = mean_squared_error(y6_test_ec, y_ec_pred)
print("MSE test: {}".format(mse_ec))
print("-------------------------------------")


# In[151]:


#Decision Tree Regressor
print("Decision Tree Regressor")
print("-------------------------------------")
tree=DecisionTreeRegressor()
tree.fit(X_train_ec_pca, y6_train_ec)
# Compute R^2 
R2_ec = tree.score(X_test_ec_pca, y6_test_ec)
print("R^2 test: {}".format(R2_ec))
#Compute Mean Squared Error
y_ec_pred = tree.predict(X_test_ec_pca)
mse_ec = mean_squared_error(y6_test_ec, y_ec_pred)
print("MSE test: {}".format(mse_ec))
print("-------------------------------------")


# percent change in the USHPCI 9 months ahead

# In[152]:


#Linear Regression 
print("Linear Regression")
print("-------------------------------------")
lr = LinearRegression()
lr.fit(X_train_ec_pca, y9_train_ec)

# Compute R^2 
R2_ec = lr.score(X_test_ec_pca, y9_test_ec)
print("R^2 test: {}".format(R2_ec))
#Compute Mean Squared Error
y_ec_pred = lr.predict(X_test_ec_pca)
mse_ec = mean_squared_error(y9_test_ec, y_ec_pred)
print("MSE test: {}".format(mse_ec))
print("-------------------------------------")


# In[153]:


#SVM 
print("SVM")
print("-------------------------------------")
svm = SVR()
svm.fit(X_train_ec_pca, y9_train_ec)

# Compute R^2 
R2_ec = svm.score(X_test_ec_pca, y9_test_ec)
print("R^2 test: {}".format(R2_ec))
#Compute Mean Squared Error
y_ec_pred = svm.predict(X_test_ec_pca)
mse_ec = mean_squared_error(y9_test_ec, y_ec_pred)
print("MSE test: {}".format(mse_ec))
print("-------------------------------------")


# In[154]:


#Lasso Regression 
print("Lasso Regression")
print("-------------------------------------")
lasso = Lasso()
lasso.fit(X_train_ec_pca, y9_train_ec)

# Compute R^2 
R2_ec = lasso.score(X_test_ec_pca, y9_test_ec)
print("R^2 test: {}".format(R2_ec))
#Compute Mean Squared Error
y_ec_pred = lasso.predict(X_test_ec_pca)
mse_ec = mean_squared_error(y9_test_ec, y_ec_pred)
print("MSE test: {}".format(mse_ec))
print("-------------------------------------")


# In[155]:


#Ridge Regression 
print("Ridge Regression")
print("-------------------------------------")
ridge = Ridge(normalize= True)
ridge.fit(X_train_ec_pca, y9_train_ec)

# Compute R^2 
R2_ec = ridge.score(X_test_ec_pca, y9_test_ec)
print("R^2 test: {}".format(R2_ec))
#Compute Mean Squared Error
y_ec_pred = ridge.predict(X_test_ec_pca)
mse_ec = mean_squared_error(y9_test_ec, y_ec_pred)
print("MSE test: {}".format(mse_ec))
print("-------------------------------------")


# In[156]:


#Decision Tree Regressor
print("Decision Tree Regressor")
print("-------------------------------------")
tree=DecisionTreeRegressor()
tree.fit(X_train_ec_pca, y9_train_ec)
# Compute R^2 
R2_ec = tree.score(X_test_ec_pca, y9_test_ec)
print("R^2 test: {}".format(R2_ec))
#Compute Mean Squared Error
y_ec_pred = tree.predict(X_test_ec_pca)
mse_ec = mean_squared_error(y9_test_ec, y_ec_pred)
print("MSE test: {}".format(mse_ec))
print("-------------------------------------")


# #### 4) Hyperparameter tuning

# Since Decision Tree Regressor regression performs better in all three target variables, so we are going to optimize its hyperparameters.

# Target: percent change in the USHPCI 3 months ahead

# In[157]:


warnings.filterwarnings("ignore")
max_depth = np.arange(1,30)
param_grid = {'max_depth': max_depth}
tree = DecisionTreeRegressor()
#tree_cv = RandomizedSearchCV(tree,param_grid,cv=10)
tree_cv = GridSearchCV(tree,param_grid,cv=10)
tree_cv.fit(X_train_ec_pca, y3_train_ec)
print("Tuned Decision Tree Classifier Parameter: {}".format(tree_cv.best_params_))
print("Tuned Decision Tree Classifier Accuracy: {}".format(tree_cv.best_score_))


# In[158]:


warnings.filterwarnings("ignore")
max_depth = np.arange(1,30)
param_grid = {'max_depth': max_depth}
tree = DecisionTreeRegressor()
#tree_cv = RandomizedSearchCV(tree,param_grid,cv=10)
tree_cv = GridSearchCV(tree,param_grid,cv=10)
tree_cv.fit(X_train_ec, y3_train_ec)
print("Tuned Decision Tree Regressor Parameter: {}".format(tree_cv.best_params_))
print("Tuned Decision Tree Regressor Accuracy: {}".format(tree_cv.best_score_))
depth3=int(tree_cv.best_params_['max_depth'])


# In[ ]:





# Target: percent change in the USHPCI 6 months ahead

# In[159]:


warnings.filterwarnings("ignore")
max_depth = np.arange(1,30)
param_grid = {'max_depth': max_depth}
tree = DecisionTreeRegressor()
tree_cv = GridSearchCV(tree, param_grid, cv = 10)
tree_cv.fit(X_train_ec, y6_train_ec)
print("Tuned Decision Tree Regressor Parameter: {}".format(tree_cv.best_params_))
print("Tuned Decision Tree Regressor Accuracy: {}".format(tree_cv.best_score_))
depth6 = int(tree_cv.best_params_['max_depth'])


# In[ ]:





# Target: percent change in the USHPCI 9 months ahead

# In[160]:


warnings.filterwarnings("ignore")
max_depth = np.arange(1,30)
param_grid = {'max_depth': max_depth}
tree = DecisionTreeRegressor()
tree_cv = GridSearchCV(tree, param_grid, cv = 10)
tree_cv.fit(X_train_ec, y9_train_ec)
print("Tuned Decision Tree Regressor Parameter: {}".format(tree_cv.best_params_))
print("Tuned Decision Tree Regressor Accuracy: {}".format(tree_cv.best_score_))
depth9 = int(tree_cv.best_params_['max_depth'])


# #### 5) Ensembling 

# Random Forest

# Target: percent change in the USHPCI 3 months ahead

# In[161]:


forest = RandomForestRegressor(n_estimators = 200, max_depth = depth3)         
forest.fit(X_train_ec, y3_train_ec)
importances = forest.feature_importances_
sorted_index = np.argsort(importances)[::-1]
x = range(X_train_ec.shape[1])
labels = np.array(econcycle.columns[:-3])[sorted_index]
plt.figure(figsize = (12,5))
plt.bar(x, importances[sorted_index], tick_label = labels)
plt.title("Feature Importances for percent change in the USHPCI 3 months ahead")
plt.xticks(rotation = 90)
plt.show()


# In[162]:


y_train_pred = forest.predict(X_train_ec)
y_test_pred = forest.predict(X_test_ec)
plt.figure(figsize = (10,5))
plt.scatter(y3_train_ec, y_train_pred, label = 'train')
plt.scatter(y3_test_ec, y_test_pred, label = 'test')
plt.title("train vs test scatter plot \n percent change in the USHPCI 3 months ahead")
plt.xlabel('PCT 3MO FWD', fontsize=15)
plt.ylabel('y_pred', fontsize=15)
plt.legend()
plt.show()


# Target: percent change in the USHPCI 6 months ahead

# In[163]:


forest = RandomForestRegressor(n_estimators = 200, max_depth = depth6)         
forest.fit(X_train_ec, y6_train_ec)
importances = forest.feature_importances_
sorted_index = np.argsort(importances)[::-1]
x = range(X_train_ec.shape[1])
labels = np.array(econcycle.columns[:-3])[sorted_index]
plt.figure(figsize = (12,5))
plt.bar(x, importances[sorted_index], tick_label = labels)
plt.title("Feature Importances for percent change in the USHPCI 6 months ahead")
plt.xticks(rotation = 90)
plt.show()


# In[164]:


y_train_pred = forest.predict(X_train_ec)
y_test_pred = forest.predict(X_test_ec)
plt.figure(figsize = (10,5))
plt.scatter(y6_train_ec, y_train_pred, label = 'train')
plt.scatter(y6_test_ec, y_test_pred, label = 'test')
plt.title("train vs test scatter plot \n percent change in the USHPCI 6 months ahead")
plt.xlabel('PCT 6MO FWD', fontsize=15)
plt.ylabel('y_pred', fontsize=15)
plt.legend()
plt.show()


# Target: percent change in the USHPCI 9 months ahead

# In[165]:


forest = RandomForestRegressor(n_estimators = 200, max_depth = depth9)         
forest.fit(X_train_ec, y9_train_ec)
importances = forest.feature_importances_
sorted_index = np.argsort(importances)[::-1]
x = range(X_train_ec.shape[1])
labels = np.array(econcycle.columns[:-3])[sorted_index]
plt.figure(figsize = (12,5))
plt.bar(x, importances[sorted_index], tick_label = labels)
plt.title("Feature Importances for percent change in the USHPCI 9 months ahead")
plt.xticks(rotation = 90)
plt.show()


# In[166]:


y_train_pred = forest.predict(X_train_ec)
y_test_pred = forest.predict(X_test_ec)
plt.figure(figsize = (10,5))
plt.scatter(y9_train_ec, y_train_pred, label = 'train')
plt.scatter(y9_test_ec, y_test_pred, label = 'test')
plt.title("train vs test scatter plot \n percent change in the USHPCI 9 months ahead")
plt.xlabel('PCT 9MO FWD', fontsize=15)
plt.ylabel('y_pred', fontsize=15)
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





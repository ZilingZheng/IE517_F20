#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import pylab
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.svm import SVC
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# ## Part 1: Exploratory Data Analysis

# Describe the data set sufficiently using the methods and visualizations that we used previously.  Include any output, graphs, tables, that you think is necessary to represent the data.  Label your figures and axes. 
# Split data into training and test sets.  Use random_state = 42. Use 85% of the data for the training set.

# In[2]:


#Import dataset
treasury = pd.read_csv('/Users/zilingzheng/Desktop/IE517/hw5/hw5_treasury yield curve data.csv')
treasury.head()


# In[3]:


treasury.dropna(inplace=True)
treasury.describe()


# In[4]:


#print dimension of data frame
nrow = treasury.shape[0]
ncol = treasury.shape[1]
print("Number of Rows of Data = ", nrow) 
print("Number of Columns of Data = ", ncol)


# In[5]:


#Scatterplot Matrix
sns.set(style='whitegrid', context='notebook')
cols = ['SVENF01','SVENF05','SVENF10','SVENF15','SVENF20','SVENF25','SVENF30','Adj_Close']
sns.pairplot(treasury[cols], height=2.5)
plt.show()


# In[6]:


corMat = pd.DataFrame(treasury.corr())
corMat


# In[7]:


cm = np.corrcoef(treasury[cols].values.T)
plt.figure(figsize=(8,8))
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=cols,
                 xticklabels=cols)
plt.show()


# In[8]:


#ECDF
def ecdf(data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1) / n
    return x, y

x_close, y_close = ecdf(treasury['Adj_Close'])
plt.plot(x_close, y_close, marker = '.', linestyle ='none')
plt.xlabel('Adj_Close')
plt.ylabel('ECDF')
plt.show()


# In[9]:


#Quantile‐Quantile Plot
stats.probplot(treasury['Adj_Close'],dist="norm",plot=pylab)
plt.show()


# ## Part 2: Perform a PCA on the Treasury Yield dataset

# Compute and display the explained variance ratio for all components, then recalculate and display on n_components=3.

# In[5]:


#Pick feature and label
X = treasury[treasury.columns[1:-1]].values
y = treasury['Adj_Close'].values

# Split train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state=42)

#Standardize the features
scaler= StandardScaler()
X_train_std= scaler.fit_transform(X_train)
X_test_std= scaler.transform(X_test)


# In[6]:



# Create a PCA model for all components: pca
pca = PCA()

#Transform 
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

print('Explained variance ratio for all components: ', pca.explained_variance_ratio_)
print('Explained variance for all components: ', pca.explained_variance_)


# In[7]:


# Plot the explained variances
features = range(pca.n_components_)
plt.figure(figsize=(12,5))
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()


# In[8]:


# Create a PCA model with 3 components: pca
pca3 = PCA(n_components=3)

#Transform 
X_train_pca3 = pca3.fit_transform(X_train_std)
X_test_pca3 = pca3.transform(X_test_std)

print('Explained variance ratio of the 3 component version: ', pca3.explained_variance_ratio_)
print('Explained variance of the 3 component version: ', pca3.explained_variance_)
print('The cumulative explained variance ratio of the 3 component version ', 
      pca3.explained_variance_ratio_[0]+pca3.explained_variance_ratio_[1]+pca3.explained_variance_ratio_[2])
print('The cumulative explained variance of the 3 component version ', 
      pca3.explained_variance_[0]+pca3.explained_variance_[1]+pca3.explained_variance_[2])


# In[9]:


# Plot the explained variances
features = range(pca3.n_components_)
plt.bar(features, pca3.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()


# ## Part 3: Linear regression v. SVM regressor - baseline
# 

# Fit a linear regression model to both datasets (the original dataset with 30 attributes and the PCA transformed dataset with 3 PCs.) using SKlearn.  Calculate its accuracy R2 score and RMSE for both in sample and out of sample (train and test sets).

# In[10]:


#Linear regression without PCA
print("---- Linear regression without PCA----")


#Split data into training and test sets
#Pick feature and label
X = treasury[treasury.columns[1:-1]].values
y = treasury['Adj_Close'].values
# Split train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state=42)
#Standardize the features
scaler= StandardScaler().fit(X_train)
X_train= scaler.transform(X_train)
X_test= scaler.transform(X_test)

start_reg= time.process_time()

# Create the regressor
reg = LinearRegression()
# Fit the regressor to the training data
reg.fit(X_train, y_train)

end_reg= time.process_time()

# Compute R^2 
R2_train = reg.score(X_train, y_train)
R2_test = reg.score(X_test, y_test)
print("R^2 training: {}".format(R2_train))
print("R^2 test: {}".format(R2_test))
#Compute Mean Squared Error
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
print("RMSE training: {}".format(rmse_train))
print("RMSE test: {}".format(rmse_test))
print("Training time: {}s".format(end_reg-start_reg))


# In[11]:


#Linear regression with PCA
print("---- Linear regression with PCA----")

start_reg_pca= time.process_time() 

# Create the regressor
reg_pca = LinearRegression()
# Fit the regressor to the training data
reg_pca.fit(X_train_pca3, y_train)

end_reg_pca= time.process_time()

# Compute R^2 
R2_train_pca = reg_pca.score(X_train_pca3, y_train)
R2_test_pca = reg_pca.score(X_test_pca3, y_test)
print("R^2 training: {}".format(R2_train_pca))
print("R^2 test: {}".format(R2_test_pca))
#Compute Mean Squared Error
y_train_pred_pca = reg_pca.predict(X_train_pca3)
y_test_pred_pca = reg_pca.predict(X_test_pca3)
rmse_train_pca = np.sqrt(mean_squared_error(y_train, y_train_pred_pca))
rmse_test_pca = np.sqrt(mean_squared_error(y_test, y_test_pred_pca))
print("RMSE training: {}".format(rmse_train_pca))
print("RMSE test: {}".format(rmse_test_pca))
print("Training time: {}s".format(end_reg_pca-start_reg_pca))


# Fit a SVM regressor model to both datasets using SKlearn.  Calculate its accuracy R2 score and RMSE for both in sample and out of sample (train and test sets). 

# In[16]:


#SVM without PCA
print("----SVM without PCA----")

start_svm= time.process_time()

# Create the regressor
#svm = SVR(kernel="linear")
svm = SVR(kernel="rbf")

# Fit the regressor to the training data
svm.fit(X_train, y_train)

end_svm= time.process_time() 

# Compute R^2 
R2_train_svm = svm.score(X_train, y_train)
R2_test_svm = svm.score(X_test, y_test)
print("R^2 training: {}".format(R2_train_svm))
print("R^2 test: {}".format(R2_test_svm))
#Compute Mean Squared Error
y_train_pred_svm = svm.predict(X_train)
y_test_pred_svm = svm.predict(X_test)
rmse_train_svm = np.sqrt(mean_squared_error(y_train, y_train_pred_svm))
rmse_test_svm = np.sqrt(mean_squared_error(y_test, y_test_pred_svm))
print("RMSE training: {}".format(rmse_train_svm))
print("RMSE test: {}".format(rmse_test_svm))
print("Training time: {}s".format(end_svm-start_svm))


# In[17]:


#SVM with PCA
print("----SVM with PCA----")

start_svm_pca= time.process_time() 

# Create the regressor
#svm_pca = SVR(kernel="linear")
svm_pca = SVR(kernel="rbf", gamma='auto')

# Fit the regressor to the training data
svm_pca.fit(X_train_pca3, y_train)

end_svm_pca= time.process_time() 

# Compute R^2 
R2_train_svm_pca = svm_pca.score(X_train_pca3, y_train)
R2_test_svm_pca = svm_pca.score(X_test_pca3, y_test)
print("R^2 training: {}".format(R2_train_svm_pca))
print("R^2 test: {}".format(R2_test_svm_pca))
#Compute Mean Squared Error
y_train_svm_pred_pca = svm_pca.predict(X_train_pca3)
y_test_svm_pred_pca = svm_pca.predict(X_test_pca3)
rmse_train_svm_pca = np.sqrt(mean_squared_error(y_train, y_train_svm_pred_pca))
rmse_test_svm_pca = np.sqrt(mean_squared_error(y_test, y_test_svm_pred_pca))
print("RMSE training: {}".format(rmse_train_svm_pca))
print("RMSE test: {}".format(rmse_test_svm_pca))
print("Training time: {}s".format(end_svm_pca-start_svm_pca))


# ## Part 4: Conclusions

# **Results worksheet**
# 
# |*Experiment 1 (Treasury Yields)*|linear|SVR|
# |-|-|-|
# |Baseline|Train Acc: 0.9023 |Train Acc: 0.9887|
# |(all attributes)|Test Acc: 0.9041|Test Acc: 0.9891|
# | -|-|-|
# |PCA transform|Train Acc: 0.8672|Train Acc: 0.9901|
# |(3 PCs)|Test Acc: 0.8662|Test Acc: 0.9900|
# 
# 
# |*Experiment 1 (Treasury Yields)*|linear|SVR|
# |-|-|-|
# |Baseline|Train RMSE: 0.7767 |Train RMSE: 0.2647|
# |(all attributes)|Test RMSE: 0.7824|Test RMSE: 0.2635|
# | -|-|-|
# |PCA transform|Train RMSE: 0.9053|Train RMSE: 0.2475|
# |(3 PCs)|Test RMSE: 0.9241|Test RMSE: 0.2531|

# The SVM performs better on the untransformed data since it has a smaller RMSE and the R^2 is closer to 1. After the PCA transformation, the cumulative explained variance ratio of the first three components is 99.44%, which means that the correlation coefficients for most variables in the datasets are high. By fitting both models, we can see that SVM still performs better, the RMSE is smaller than before and the R^2 slightly increases. By looking at the training time, Linear Regression is much faster than SVM. The training times for both model imporve with PCA.

# ## Part 5: Appendix

# Link to GitHub: https://github.com/ZilingZheng/IE517_F20/blob/master/IE517_FA20_HW5.py

# In[19]:


print("My name is Ziling Zheng")
print("My NetID is: zzheng27")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


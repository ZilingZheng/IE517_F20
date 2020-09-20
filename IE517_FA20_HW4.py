#!/usr/bin/env python
# coding: utf-8

# In[26]:


import sys
import pylab
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge


# ## Part 1: Exploratory Data Analysis

# Describe the data sufficiently using the methods and visualizations that we used previously in Module 3 and again this week. 
# Include any output, graphs, tables, heatmaps, box plots, etc.  Label your figures and axes. 

# In[2]:


#Import dataset
housing = pd.read_csv('/Users/zilingzheng/Desktop/IE517/hw4/housing.csv')
housing.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                   'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

housing.head()


# In[3]:


#print dimension of data frame
nrow = housing.shape[0]
ncol = housing.shape[1]
print("Number of Rows of Data = ", nrow) 
print("Number of Columns of Data = ", ncol)


# In[4]:


#print summary of data frame
housing.describe()


# In[5]:


#Scatterplot Matrix
sns.set(style='whitegrid', context='notebook')
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(housing[cols], height=2.5)
plt.show()


# In[6]:


corMat = pd.DataFrame(housing.corr())
corMat


# In[7]:


#ECDF
def ecdf(data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1) / n
    return x, y

x_MEDV, y_MEDV = ecdf(housing['MEDV'])
plt.plot(x_MEDV, y_MEDV, marker = '.', linestyle ='none')
plt.xlabel('MEDV')
plt.ylabel('ECDF')
plt.show()


# In[8]:


#Quantile‐Quantile Plot
stats.probplot(housing['MEDV'],dist="norm",plot=pylab)
plt.show()



# ## Part 2: Linear regression

# Fit a linear model using SKlearn to all of the features of the dataset. 
# Describe the model (coefficients and y intercept), plot the residual errors,
# calculate performance metrics: MSE and R2.

# In[9]:


housing2 = pd.read_csv('/Users/zilingzheng/Desktop/IE517/hw4/housing2(2).csv')
housing2.head()


# In[10]:


#Split data into training and test sets

#Pick feature and label
X = housing2[housing.columns[:-1]].values
y = housing2['MEDV'].values

# Split train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

#Standardize the features
scaler= preprocessing.StandardScaler().fit(X_train)
X_train= scaler.transform(X_train)
X_test= scaler.transform(X_test)


# In[11]:


# Create the regressor
reg = LinearRegression()
# Fit the regressor to the training data
reg.fit(X_train, y_train)
#coefficients
coef = reg.coef_
print('coefficient: ', coef)
#y-intercept
y_inter = reg.intercept_
print('y-intercept: ', y_inter)


# In[12]:


#Plot the residual errors
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o',label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test, c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted MEDV')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='red', lw=2)
plt.xlim([-10, 50])
plt.show()


# In[13]:


# Compute R^2 
R2_train = reg.score(X_train, y_train)
R2_test = reg.score(X_test, y_test)
print("R^2 training: {}".format(R2_train))
print("R^2 test: {}".format(R2_test))
#Compute Mean Squared Error
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
print("MSE training: {}".format(mse_train))
print("MSE test: {}".format(mse_test))


# In[ ]:





# ## Part 3 Penalized Linear Regression
# 
# 
# #### 1: Ridge regression
# 

# Fit a Ridge model using SKlearn to all of the features of the dataset.
# Test some settings for alpha. Describe the model (coefficients and y intercept),
# plot the residual errors, calculate performance metrics: MSE and R2. 
# Which alpha gives the best performing model?
# 

# In[14]:


# Create a ridge regressor
ridge = Ridge(normalize= True)

# Setup the array of alphas and lists to store scores
alpha_space_ridge = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1])
coef_ridge=[]
y_inter_ridge=[]
mse_train_ridge=[]
mse_test_ridge=[]
r2_train_ridge=[]
r2_test_ridge=[]

for alpha in alpha_space_ridge:
    ridge.alpha= alpha
    ridge.fit(X_train, y_train)
    #coefficients
    coef_ridge.append(ridge.coef_)
    #y-intercept
    y_inter_ridge.append(ridge.intercept_)
    #mse
    y_train_ridge_pred=ridge.predict(X_train)
    y_test_ridge_pred=ridge.predict(X_test)
    mse_train_ridge.append(mean_squared_error(y_train,y_train_ridge_pred))
    mse_test_ridge.append(mean_squared_error(y_test,y_test_ridge_pred))
    #R^2
    r2_train_ridge.append(ridge.score(X_train, y_train))
    r2_test_ridge.append(ridge.score(X_test, y_test))    

print('Pick alpha :{}'.format(alpha_space_ridge))
print('coefficient for all alpha: {}'.format(coef_ridge))
print('y-intercept for all alpha: {}'.format(y_inter_ridge))


# In[15]:


#Find the best alpha
index_ridge = mse_test_ridge.index(min(mse_test_ridge))
best_alpha_ridge = alpha_space_ridge[index_ridge]
print('The best alpha: ', best_alpha_ridge)
#coefficient for best alpha
print('coefficient for best alpha: ', coef_ridge[index_ridge])
#y-intercept for best alpha
print('y-intercept for best alpha: ', y_inter_ridge[index_ridge])


# In[16]:


#Plot the residual errors for best alpha
ridge_best = Ridge(alpha = alpha_space_ridge[index_ridge])
ridge_best.fit(X_train, y_train)
y_train_best_ridge_pred = ridge_best.predict(X_train)
y_test_best_ridge_pred = ridge_best.predict(X_test)
plt.scatter(y_train_best_ridge_pred, y_train_best_ridge_pred - y_train, c='blue', marker='o',label='Training data')
plt.scatter(y_test_best_ridge_pred,  y_test_best_ridge_pred - y_test, c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted MEDV')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='red', lw=2)
plt.xlim([-10, 50])
plt.show()


# In[17]:


#R^2
print("R^2 training for all alpha: {}".format(r2_train_ridge))
print("R^2 test for all alpha: {}".format(r2_test_ridge))
#Mean Squared Error
print("MSE training for all alpha: {}".format(mse_train_ridge))
print("MSE test for all alpha: {}".format(mse_test_ridge))


# In[18]:


#R^2 for best alpha
print("R^2 training for best alpha: {}".format(r2_train_ridge[index_ridge]))
print("R^2 test for best alpha: {}".format(r2_test_ridge[index_ridge]))
#Mean Squared Error for best alpha
print("MSE training for best alpha: {}".format(mse_train_ridge[index_ridge]))
print("MSE test for best alpha: {}".format(mse_test_ridge[index_ridge]))


# In[ ]:





# #### 2: LASSO regression

# In[19]:


# Create a lasso regressor
lasso = Lasso()

# Setup the array of alphas and lists to store scores
alpha_space_lasso = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1])
coef_lasso=[]
y_inter_lasso=[]
mse_train_lasso=[]
mse_test_lasso=[]
r2_train_lasso=[]
r2_test_lasso=[]

for alpha in alpha_space_lasso:
    lasso.alpha= alpha
    lasso.fit(X_train, y_train)
    #coefficients
    coef_lasso.append(lasso.coef_)
    #y-intercept
    y_inter_lasso.append(lasso.intercept_)
    #mse
    y_train_lasso_pred=lasso.predict(X_train)
    y_test_lasso_pred=lasso.predict(X_test)
    mse_train_lasso.append(mean_squared_error(y_train,y_train_lasso_pred))
    mse_test_lasso.append(mean_squared_error(y_test,y_test_lasso_pred))
    #R^2
    r2_train_lasso.append(lasso.score(X_train, y_train))
    r2_test_lasso.append(lasso.score(X_test, y_test))   
    
    
print('Pick alpha :{}'.format(alpha_space_lasso))
print('coefficient for all alpha: {}'.format(coef_lasso))
print('y-intercept for all alpha: {}'.format(y_inter_lasso))


# In[20]:


#Find the best alpha
index_lasso = mse_test_lasso.index(min(mse_test_lasso))
best_alpha_lasso = alpha_space_lasso[index_lasso]
print('The best alpha: ', best_alpha_lasso)
#coefficient for best alpha
print('coefficient for best alpha: ', coef_lasso[index_lasso])
#y-intercept for best alpha
print('y-intercept for best alpha: ', y_inter_lasso[index_lasso])


# In[21]:


#Plot the residual errors for best alpha
lasso_best = Lasso(alpha = alpha_space_lasso[index_lasso])
lasso_best.fit(X_train, y_train)
y_train_best_lasso_pred = lasso_best.predict(X_train)
y_test_best_lasso_pred = lasso_best.predict(X_test)
plt.scatter(y_train_best_lasso_pred, y_train_best_lasso_pred - y_train, c='blue', marker='o',label='Training data')
plt.scatter(y_test_best_lasso_pred,  y_test_best_lasso_pred - y_test, c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted MEDV')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='red', lw=2)
plt.xlim([-10, 50])
plt.show()


# In[22]:


#R^2
print("R^2 training for all alpha: {}".format(r2_train_lasso))
print("R^2 test for all alpha: {}".format(r2_test_lasso))
#Mean Squared Error
print("MSE training for all alpha: {}".format(mse_train_lasso))
print("MSE test for all alpha: {}".format(mse_test_lasso))


# In[23]:


#R^2 for best alpha
print("R^2 training for best alpha: {}".format(r2_train_lasso[index_lasso]))
print("R^2 test for best alpha: {}".format(r2_test_lasso[index_lasso]))
#Mean Squared Error for best alpha
print("MSE training for best alpha: {}".format(mse_train_lasso[index_lasso]))
print("MSE test for best alpha: {}".format(mse_test_lasso[index_lasso]))


# ## Part 4: Conclusions

# Write a short paragraph summarizing your findings.  

# I fit the housing data into three models: Linear regression, Ridge regression, and Lasso regression. By comparing the mean squared error for test sets of data, we get Linear(24.2911) < Lasso(24.2945) < Ridge(24.2999), therefore, the Linear regression has the best performing on this data.
# Also, for both Ridge and Lasso regression, we can see that when the alpha is getting closer to zero, the mse for test sets of data is smaller and the models perform better.

# In[ ]:





# ## Part 5: Appendix
# https://github.com/ZilingZheng/IE517_F20/edit/master/IE517%20HW4.py

# 

# In[ ]:





# In[24]:


print("My name is Ziling Zheng")
print("My NetID is: zzheng27")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


# In[ ]:





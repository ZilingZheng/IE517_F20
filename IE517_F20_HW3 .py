#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pylab
import numpy as np
import pandas as pd
from pandas import DataFrame
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt




# In[2]:


HY_corporate_bond = pd.read_csv('/Users/zilingzheng/Desktop/IE517/hw3/HY_Universe_corporate bond.csv')

#print head and tail of data frame
print(HY_corporate_bond.head())
print(HY_corporate_bond.tail())


# In[3]:


#print dimension of data frame
nrow = HY_corporate_bond.shape[0]
ncol = HY_corporate_bond.shape[1]
print("Number of Rows of Data = ", HY_corporate_bond.shape[0]) 
print("Number of Columns of Data = ", HY_corporate_bond.shape[1])


# In[4]:


#print summary of data frame
summary = HY_corporate_bond.describe()
print(summary)


# In[5]:


#Histogram
sns.set()
plt.hist(HY_corporate_bond['LIQ SCORE'],bins=20)
plt.xlabel('LIQ SCORE')
plt.ylabel('count')
plt.show()


# In[6]:


#Bee Swarm Plot
sns.set()
_ = sns.swarmplot(x='Industry', y='Issued Amount', data=HY_corporate_bond)

_ = plt.xlabel('Industry')
_ = plt.ylabel('Issued Amount')

plt.show()


# In[7]:



sns.set()
_ = sns.swarmplot(x='bond_type', y='LIQ SCORE', data=HY_corporate_bond)

_ = plt.xlabel('bond_type')
_ = plt.ylabel('LIQ SCORE')

plt.show()


# In[8]:


#ECDF
def ecdf(data):
    # Number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x = np.sort(data)
    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n
    return x, y

#x_vers, y_vers = ecdf(HY_corporate_bond['LIQ SCORE'])
x_jhk, y_jhk = ecdf(HY_corporate_bond['Months in JNK'])
x_hyg, y_hyg = ecdf(HY_corporate_bond['Months in HYG'])
x_both, y_both = ecdf(HY_corporate_bond['Months in Both'])

plt.plot(x_jhk, y_jhk, marker = '.', linestyle ='none')
plt.plot(x_hyg, y_hyg, marker = '.', linestyle ='none')
plt.plot(x_both, y_both, marker = '.', linestyle ='none')

plt.legend(('JHK', 'HYG', 'Both'), loc='lower right')
plt.xlabel('Months')
plt.ylabel('ECDF')

plt.show()


# In[9]:


#scatter Plot
plt.scatter(HY_corporate_bond['Months in JNK'],HY_corporate_bond['LIQ SCORE'])
plt.scatter(HY_corporate_bond['Months in HYG'],HY_corporate_bond['LIQ SCORE'])
#plt.scatter(HY_corporate_bond['Months in Both'],HY_corporate_bond['LIQ SCORE'])


# In[10]:


#Quantile‐Quantile Plot
from scipy import stats 
stats.probplot(HY_corporate_bond['LIQ SCORE'], dist="norm",plot=pylab)
plt.show()


# In[11]:


#Box Plot

sns.boxplot(x='bond_type',y='LIQ SCORE', data=HY_corporate_bond)
plt.xlabel('bond_type')
plt.ylabel('LIQ SCORE')
plt.show()


# In[12]:


#Parallel coordinates
from pandas.plotting import parallel_coordinates
data = HY_corporate_bond.iloc[:200,5:9]
parallel_coordinates(data,'S_and_P')
plt.show()


# In[13]:


#Cross Plot

sp = HY_corporate_bond.iloc[:200,6]
fitch = HY_corporate_bond.iloc[:200,7]
plt.scatter(sp, fitch)
plt.xlabel("S and P") 
plt.ylabel(("Fitch"))
plt.show()


# In[14]:


#Heat Map

corMat = DataFrame(HY_corporate_bond.corr())
#visualize correlations using heatmap
plt.pcolor(corMat)
plt.show()


# In[15]:


print("My name is Ziling Zheng")
print("My NetID is: zzheng27")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


# In[ ]:





# In[ ]:





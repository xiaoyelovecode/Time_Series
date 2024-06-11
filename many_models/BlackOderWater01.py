#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Uploading required libraries
import pandas as pd
from pandas import DataFrame
import numpy as np
import sklearn
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Uploaoding the interactiveshell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[3]:


# Read in the data file
DATAPATH = 'C:/Users/yxie/TEACH/Geog670/Fall2023/Week11/FinalDataforLassoN_F3.csv'
data = pd.read_csv(DATAPATH)
data.head()


# In[4]:


# Arrange the data for analysis
data.columns
x = data.drop(['FAC3'], axis=1)
y = data['FAC3']
x.head()
y.head()


# In[5]:


# Split the data in the training and testing
train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=0.15, random_state=1)
train_x.shape
test_x.shape
train_y.shape
test_y.shape


# In[6]:


# Run Linear Regression and lasso Regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
lm = LinearRegression()
lm_lasso = Lasso(alpha=0.01)
lm
lm_lasso

lm.fit(train_x, train_y)
lm_lasso.fit(train_x, train_y)


# In[7]:


# Plot the linear model rsult
plt.figure(figsize=(15,10))
ft_importances_lm = pd.Series(lm.coef_, index = x.columns)
ft_importances_lm.plot(kind ='barh')
plt.show()
# print(lm.coef_, x.columns)


# In[9]:


# Write out the linear model results
columns=x.columns
dictionary={} 
dictionary["xvars"]=columns
dictionary[ "coeff" ]=list(lm.coef_)
saveName='Regress_Out1.xlsx'
savePath='C:/Users/yxie/TEACH/Geog670/Fall2023/Week11/'+saveName
writer=pd.ExcelWriter(savePath)
df=DataFrame(dictionary)
df.to_excel(writer,'Sheet1')
writer.close()


# In[10]:


# Plot the lasso regression results (significant bands)
plt.figure(figsize=(15,10))
ft_importances_lm_lasso = pd.Series(lm_lasso.coef_, index = x.columns)
ft_importances_lm_lasso.plot(kind ='barh')
plt.show()

# Write out the lasso significant bands
columns=x.columns
dictionary={} 
dictionary["xvars"]=columns
dictionary[ "coeff" ]=list(lm_lasso.coef_)
saveName='Lasso_F3_A001_CV15.xlsx'
savePath='C:/Users/yxie/TEACH/Geog670/Fall2023/Week11/'+saveName
writer=pd.ExcelWriter(savePath)
df=DataFrame(dictionary)
df.to_excel(writer,'Sheet1')
writer.close()


# In[12]:


print("RSquare Value for Simple Regression TEST data is-")
np.round(lm.score(test_x,test_y)*100,2)
print("RSquare Value for Lasso Regression TEST data is-")
np.round(lm_lasso.score(test_x,test_y)*100,2)


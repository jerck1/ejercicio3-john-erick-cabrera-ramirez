#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.linear_model

get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


data = np.loadtxt("notas_andes.dat", skiprows=1)


# In[5]:


Y = data[:,4]
X = data[:,:4]


# In[35]:


regresion = sklearn.linear_model.LinearRegression()


# In[41]:


x_ran=np.ones(69)
y_ran=np.ones(69)
#x_ran=np.random.choice(X[:,],69)
for i in range(0,2):
    x_ran=np.random.choice(X[:,0],69)
    y_ran=np.random.choice(Y,69)
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(x_ran, y_ran, test_size=1)
    #print(np.shape(Y_train), np.shape(X_train))
    regresion = sklearn.linear_model.LinearRegression()
    regresion.fit(X_train, Y_train)


# In[ ]:


regresion.fit(X_train, Y_train)


# In[38]:


y_ran=np.random.choice(Y,69)


# In[ ]:





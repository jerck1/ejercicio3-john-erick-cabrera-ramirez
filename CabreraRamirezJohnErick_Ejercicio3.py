#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.linear_model
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = np.loadtxt("notas_andes.dat", skiprows=1)


# In[3]:


Y = data[:,4]
X = data[:,:4]


# In[4]:


regresion = sklearn.linear_model.LinearRegression()


# In[95]:


x_ran=X
y_ran=np.ones(69)
beta=np.zeros((1000,5))
#for i in range(0,3):
i=0
for j in range(0,1000):
        x_ran[:,i]=np.random.choice(X[:,i],69)
        y_ran=np.random.choice(Y,69)
        X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(x_ran, y_ran, test_size=1)
#       print(np.shape(Y_train), np.shape(X_train), np.shape(X_test), np.shape(Y_test))
        regresion = sklearn.linear_model.LinearRegression()
        regresion.fit(X_train, Y_train)
#        print(regresion.coef_, regresion.intercept_)
#        print(regresion.score(X_train,Y_train))
#        for k in range()
        beta[j][0]=regresion.intercept_
        beta[j][1:5]=regresion.coef_
#        labels = ["$\beta_0$","$\beta_1$", "$\beta_2$", "$\beta_3$", "$\beta_4$"]
   # x = [21,22,23,4,5,6,77,8,9,10,31,32,33,34,35,36,37,18,49,50,100]
labels = ["b0", "b1", "b2", "b3","b4"]
#for k in range(5):
for k, label in enumerate(labels):
    plt.figure(figsize=(3,3))
    plt.subplot(3,2,k+1)
    num_bins = 5
    n, bins, patches = plt.hist(beta[:,k], num_bins, facecolor='blue', alpha=0.5)
    plt.xlabel(label)
    plt.show()
  #  print(k)


# In[55]:


beta[0]


# In[56]:


beta[1]


# In[64]:


beta[:,0]


# In[59]:


beta


# In[ ]:






# coding: utf-8

# In[12]:


import pandas as pd
from sklearn.datasets import load_iris 
iris = load_iris() 
data = pd. DataFrame(iris.data, columns=iris.feature_names) 
label = pd.DataFrame(list(map(lambda x : iris.target_names[x], iris.target)),columns=['Species'])
iris = pd.concat([data, label], axis=1)
print(iris.head())


# In[20]:


import seaborn as sns
type(iris)


# In[30]:


a=iris.drop(columns=['Species'])
a.head(5)


# In[29]:


sns.distplot(a["sepal length (cm)"], color='r')


# In[31]:


sns.distplot(a["sepal width (cm)"], color='r')


# In[32]:


sns.distplot(a["petal length (cm)"], color='r')


# In[33]:


sns.distplot(a["petal width (cm)"], color='r')


# In[34]:


sns.pairplot(a)


# In[79]:


import matplotlib.pyplot as plt
import numpy as np
fig, axes=plt.subplots(2,2)
plt.subplot(2,2,1)
sns.distplot(a["sepal length (cm)"])
plt.subplot(2,2,2)
sns.distplot(a["sepal width (cm)"])
plt.subplot(2,2,3)
sns.distplot(a["petal length (cm)"])
plt.subplot(2,2,4)
sns.distplot(a["petal width (cm)"])
    


# In[88]:


fig, axes=plt.subplots(2,2)

sns.boxplot(a["sepal length (cm)"],ax=axes[0, 0])
sns.boxplot(a["sepal width (cm)"],ax=axes[0, 1])
sns.boxplot(a["petal length (cm)"],ax=axes[1, 0])
sns.boxplot(a["petal width (cm)"],ax=axes[1, 1])
    


# In[65]:


sns.countplot(iris["Species"])


# In[71]:


sns.lmplot("sepal length (cm)","petal length (cm)",iris,hue='Species')


# In[90]:


sns.barplot("Species","petal length (cm)",data=iris)


# In[76]:


sns.heatmap(a)


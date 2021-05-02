#!/usr/bin/env python
# coding: utf-8

# 
# 

# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# 

# 

# In[39]:


df = pd.read_csv('College_Data',index_col=0)


# 

# In[40]:


df.head()


# 

# In[ ]:





# In[ ]:





# 

# In[41]:


sns.lmplot(x='Room.Board',y='Grad.Rate',hue='Private',data=df,fit_reg=False,size=6)


# 

# In[42]:


sns.lmplot(x='Outstate',y='F.Undergrad',hue='Private',data=df,fit_reg=False,size=6)


# 

# In[43]:


sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)


# 

# In[44]:


sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)


# 

# In[45]:


df[df['Grad.Rate']>100]


# 

# In[48]:


df['Grad.Rate']['Cazenovia College'] = 100


# In[ ]:





# In[95]:





# 

# In[49]:


from sklearn.cluster import KMeans


# 

# In[50]:


Km=KMeans(n_clusters=2)


# 

# In[54]:


Km.fit(df.drop('Private',axis=1))


# 

# In[57]:


Km.cluster_centers_


# 

# In[61]:


def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0


# In[62]:


df['cluster']=df['Private'].apply(converter)


# In[122]:





# ** Create a confusion matrix and classification report to see how well the Kmeans clustering worked without being given any labels.**

# In[65]:


from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(df['cluster'],Km.labels_))
print(classification_report(df['cluster'],Km.labels_))


# 

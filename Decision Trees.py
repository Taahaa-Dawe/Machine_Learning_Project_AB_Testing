#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
import matplotlib.pyplot as plt
import graphviz 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# In[2]:


import os
os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'


# In[39]:


Data = pd.read_csv("Wholedata.csv")


# In[40]:


Data.head()


# In[87]:


Data = Data.drop(["Date"], axis=1)


# In[88]:


Data.dropna(inplace=True)


# In[89]:


Data.head()


# In[90]:


TrainDF, TestDF = train_test_split(Data, test_size=0.30)
TestDF.head()


# In[91]:


TrainLabels=TrainDF["Campaign Name"]


# In[92]:


TrainDF = TrainDF.drop(["Campaign Name"], axis=1)


# In[93]:


TestLabels=TestDF["Campaign Name"]


# In[94]:


TestDF = TestDF.drop(["Campaign Name"], axis=1)


# In[95]:


TrainDF.head()


# In[96]:


TestDF.head()


# In[97]:


Features = TestDF.columns.tolist()


# In[98]:


MyDT=DecisionTreeClassifier(criterion='entropy', 
                            splitter='best')


# In[109]:


t = MyDT.fit(TrainDF, TrainLabels)
#MyDT.fit(eval(temp1), eval(temp2))
    ## plot the tree
plt.figure(figsize=(100,30))

plot_tree(t,filled=True,fontsize=60,feature_names=Features,class_names=["Control Campaign","Test Campaign"])
plt.savefig("Image3.png")
plt.show()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[18]:


# The Iris Setosa
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)


# In[19]:


# The Iris Versicolor
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
Image(url,width=300, height=300)


# In[20]:


# The Iris Virginica
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
Image(url,width=300, height=300)


# In[21]:


import seaborn as sns
df = sns.load_dataset('iris')


# In[22]:


df.head()


# In[37]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[53]:


#plot on bases of petal


sns.set_style("whitegrid")
sns.FacetGrid(df,hue="species",size=6)    .map(plt.scatter,"petal_length","petal_width")    .add_legend()
plt.show()


# In[76]:


# Setosa is the most separable. 
sns.set_style("whitegrid")
sns.pairplot(df,hue='species',palette="BuPu")


# In[56]:


setosa = df[df['species']=='setosa']
sns.set_style("whitegrid")
sns.kdeplot( setosa['sepal_width'], setosa['sepal_length'],
                 cmap="plasma", shade=True, shade_lowest=False, size = 10)


# In[40]:


from sklearn.model_selection import train_test_split
X = df.drop('species',axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# In[41]:


from sklearn.svm import SVC


# In[42]:


svc_model = SVC()


# In[43]:


svc_model.fit(X_train,y_train)


# In[44]:


pred = svc_model.predict(X_test)


# In[45]:


from sklearn.metrics import classification_report,confusion_matrix


# In[46]:


cm = confusion_matrix(y_test,pred)


# In[50]:


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="BuPu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[51]:


print(classification_report(y_test,pred))


# In[80]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, pred))


# In[ ]:





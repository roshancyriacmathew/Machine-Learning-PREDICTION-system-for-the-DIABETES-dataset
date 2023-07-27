#!/usr/bin/env python
# coding: utf-8

# In[1]:


# file name: diabetes.csv


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import style
style.use('seaborn')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv('diabetes.csv')


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.shape


# In[7]:


df.describe()


# In[8]:


sns.countplot(x='Outcome', data=df)


# In[9]:


plt.figure(figsize=(10,7))
sns.heatmap(df.corr(), annot=True)
plt.title('Correlation between the columns')
plt.show()


# In[10]:


df.corr()['Outcome'].sort_values()


# In[11]:


X = df.drop(columns= 'Outcome', axis=1)
Y = df['Outcome']


# In[12]:


X.hist(figsize=(10,8), bins=20)
plt.tight_layout()
plt.show()


# In[13]:


scaler = StandardScaler()


# In[14]:


scaler.fit(X)


# In[15]:


standardized_data = scaler.transform(X)


# In[16]:


X = standardized_data
Y = df['Outcome']


# In[17]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)


# In[18]:


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[19]:


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
logreg_pred = logreg.predict(X_test)
logreg_acc = accuracy_score(logreg_pred, Y_test)
print("Test Accuracy: {:.2f}%".format(logreg_acc*100))


# In[20]:


print(classification_report(Y_test, logreg_pred))


# In[21]:


svmModel = svm.SVC(kernel='linear')
svmModel.fit(X_train, Y_train)
svmModel_pred = svmModel.predict(X_test)
svmModel_acc = accuracy_score(svmModel_pred, Y_test)
print("Test Accuracy: {:.2f}%".format(svmModel_acc*100))


# In[22]:


print(classification_report(Y_test, svmModel_pred))


# In[23]:


input_data = (5,166,72,19,175,25.8,0.587,51)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
scalar_data = scaler.transform(input_data_reshaped)


# In[24]:


prediction = svmModel.predict(scalar_data)
print(prediction)


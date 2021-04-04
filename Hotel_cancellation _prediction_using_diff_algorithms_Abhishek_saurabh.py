#!/usr/bin/env python
# coding: utf-8

# # Hotel Cancellation Prediction using differnt algorithm

# In[ ]:


# Abhishek Saurabh (created by)


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


hotel = pd.read_csv(r"hotel_bookings.csv")


# In[5]:


hotel.head()


# In[6]:


hotel.describe()


# In[7]:


hotel.info()


# In[8]:


hotel.isnull().sum()


# In[9]:


hotel= hotel.drop(['company'],axis=1)    #its has maximum null value thats why removing it.


# In[11]:


hotel= hotel.dropna(axis=0)                 # Removing all the rows which have missing values


# In[12]:


hotel.info()


# In[13]:


hotel.isnull().sum()                  #agian checking for any null value


# In[14]:


hotel['hotel'].unique()      #checking the unique value in hotel


# In[15]:


categorical_features = ['hotel','is_canceled','arrival_date_week_number','meal','country','market_segment',
                        'distribution_channel','is_repeated_guest','reserved_room_type','assigned_room_type',
                        'deposit_type','agent','customer_type','reservation_status','arrival_date_month']


# In[16]:


hotel[categorical_features]=hotel[categorical_features].astype('category')


# In[17]:


hotel.info()


# In[18]:


hotel['meal'].unique()


# In[19]:


y=hotel['is_canceled']                  # seperating the data set into feature and target value


# In[20]:


y


# In[21]:


X = hotel.drop(['is_canceled','reservation_status_date'],axis=1)


# In[23]:


X


# In[24]:


X_dum=pd.get_dummies(X,prefix_sep='-',drop_first=True) # converting  the data into dummy variable


# In[25]:


X_dum


# In[ ]:


#Splitting the data into train and test


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


X_train,X_test,y_train,y_test= train_test_split(X_dum,y, test_size=.25,random_state=40)


# In[28]:


X_train


# In[29]:


from sklearn.linear_model import LogisticRegression            # making logistic regression model


# In[30]:


logistic=LogisticRegression()


# In[31]:


logistic.fit(X_train,y_train)


# In[32]:


y_pred= logistic.predict(X_test)           # #predicting the test data


# In[33]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[34]:


accuracy_score(y_test,y_pred)                   # Here is our predicted value 


# In[38]:


# Now using different algorithms to find accuracy


# In[39]:


classification_report(y_test,y_pred)


# In[40]:


#calculating the ROC and AUC  for the logistics regression


# In[41]:


from sklearn.metrics import roc_curve,roc_auc_score


# In[42]:


roc_curve(y_test,y_pred)


# In[43]:


roc_auc_score(y_test,y_pred)


# In[44]:


# Using Random forest algorithm


# In[45]:


from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier


# In[46]:


rand=RandomForestClassifier(n_jobs=10, random_state=40)


# In[47]:


gb=GradientBoostingClassifier(random_state=50)


# In[48]:


rand.fit(X_train,y_train)


# In[49]:


gb.fit(X_train,y_train)


# In[50]:


# predicting the test sample for randomforest and gradient boosting 
rand_pred=rand.predict(X_test)


# In[51]:


gb_pred=gb.predict(X_test)


# In[52]:


accuracy_score(y_test,rand_pred)


# In[53]:


accuracy_score(y_test,gb_pred)


# In[54]:


classification_report(y_test,rand_pred)


# In[55]:


classification_report(y_test,gb_pred)


# In[56]:


roc_auc_score(y_test,rand_pred)


# In[57]:


roc_auc_score(y_test,gb_pred)


# In[ ]:


#Using  confusion matrix for logistic reression,random forest and gradient boosting


# In[58]:


from sklearn.metrics import confusion_matrix


# In[59]:


confusion_matrix(y_test,y_pred)


# In[60]:


confusion_matrix(y_test,rand_pred)


# In[61]:


confusion_matrix(y_test,gb_pred)


# #  from all three algorithm we will found that random forest and gradient boosting gives maximum accuracy

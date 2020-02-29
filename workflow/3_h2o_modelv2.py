#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Working with AutoML (H2O)

Having cleaned our model,


# In[41]:


import h2o
h2o.init()
from h2o.estimators import deeplearning
from h2o.estimators import H2ORandomForestEstimator
from h2o.automl import H2OAutoML
import pandas as pd
dum_train = h2o.import_file('../cleaned-data/02_diamonds_train_alldummies.csv')
dum_test = h2o.import_file('../cleaned-data/02_diamonds_test_alldummies.csv')


# In[2]:


df = dum_train.as_data_frame()
df = df.reset_index()
df.rename(columns={'index':'id'})


# In[3]:


deep_v1 = deeplearning.H2ODeepLearningEstimator()


# In[4]:


train_1, test_1 = dum_train.split_frame([0.8])


# In[5]:


deep_v1.train(dum_train.col_names[:-1], y='price', training_frame=train_1, validation_frame=test_1)


# In[6]:


deep_v1


# In[7]:


diam_forest_v1 = H2ORandomForestEstimator(
    ntrees=10,
    stopping_rounds=2,
    score_each_iteration=True,
    seed=1000000)


# In[8]:


diam_forest_v1.train(dum_train.col_names[:-1], y='price', training_frame=train_1, validation_frame=test_1)


# In[9]:


diam_forest_v1


# In[11]:


# aml = H2OAutoML(max_runtime_secs = 60, seed = 1, project_name = "diamonds")
# aml.train(y = 'price', training_frame = train_1, leaderboard_frame = test_1)


# In[12]:


aml.leaderboard


# In[13]:


dum_train.head()


# In[42]:


aml2 = H2OAutoML(max_runtime_secs = 600, seed = 1, project_name = "diamonds", keep_cross_validation_models=True)
aml2.train(y = 'price', training_frame = dum_train)


# In[ ]:


aml2.leaderboard


# In[ ]:


pred_leader = aml2.leader.predict(dum_test)


# In[ ]:


pred_df = pred_leader.as_data_frame()
pred_df = pred_df.reset_index()
pred_df = pred_df.rename(columns={'index':'id', 'predict':'price'})


# In[ ]:


pred_df[['id','price']].to_csv('../predictions/03-Auto-ML-v1.csv', index=False)


# In[43]:


aml3 = H2OAutoML(max_runtime_secs = 600, seed = 1, project_name = "diamonds", keep_cross_validation_models=True)
aml3.train(y = 'price', training_frame = dum_train)


# In[44]:


pred_leader = aml3.leader.predict(dum_test)


# In[45]:


pred_df = pred_leader.as_data_frame()
pred_df = pred_df.reset_index()
pred_df = pred_df.rename(columns={'index':'id', 'predict':'price'})


# In[46]:


pred_df[['id','price']].to_csv('../predictions/04-Auto-ML-v2.csv', index=False)


# In[47]:


aml3.leaderboard


# ## Applying AutoML to the model without dummies
# 
# Next, we can use this algorithm on the dataset we cleaned differently, with the categorical variables as integers going up and down, to see if we get better results than with earlier models (notebook 2)

# In[54]:


int_train = h2o.import_file('../cleaned-data/01-diamonds-train-cl.csv').drop(index=[0], axis=1)
int_test = h2o.import_file('../cleaned-data/01-diamonds-test-cl.csv').drop(index=[0], axis=1)
display(int_train.head())
display(int_test.head())


# In[61]:


aml4 = H2OAutoML(max_runtime_secs = 180, seed = 1, project_name = "diamonds", keep_cross_validation_models=True)
aml4.train(y = 'price', training_frame = dum_train)


# In[62]:


pred_leader2 = aml4.leader.predict(dum_test)


# In[63]:


pred_df2 = pred_leader2.as_data_frame()
pred_df2 = pred_df2.reset_index()
pred_df2 = pred_df2.rename(columns={'index':'id', 'predict':'price'})


# In[64]:


pred_df2[['id','price']].to_csv('../predictions/05-Auto-ML-v3-without-dummies.csv', index=False)


# In[65]:


aml4.leaderboard


# In[ ]:





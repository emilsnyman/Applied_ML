#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os


# In[2]:


data = pd.read_pickle("appml-assignment1-dataset.pkl")


# In[3]:


exchange_data = pd.DataFrame(data['X'])
col_to_drop = []
for col in exchange_data.columns:
    if '-high' not in col:
        col_to_drop.append(col)
        
#print(col_to_drop)
exchange_data = exchange_data.drop(col_to_drop,axis=1)
#print(exchange_data.columns)
exchange_data = exchange_data.reset_index()
labels = pd.DataFrame(data['y'])
labels = labels.reset_index()
exchange_data['labels'] = labels['CAD-high']
#exchange_data = exchange_data.drop('date',axis=1)
#exchange_data = exchange_data.join
#print(exchange_data)
print(exchange_data.columns)


# In[4]:


from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(exchange_data,test_size=0.2,random_state=42)
exchange_tr = train_set.copy()
exchange_test=test_set.copy()


# In[5]:


exchange_tr_labels=exchange_tr["labels"].copy()
exchange_tr=exchange_tr.drop("labels",axis=1)

exchange_test_labels=exchange_test["labels"].copy()
exchange_test=exchange_test.drop("labels",axis=1)


# In[6]:


sample_incomplete_rows = exchange_data[exchange_data.isnull().any(axis=1)]
sample_incomplete_rows.head()


# In[7]:


exchange_tr_num = exchange_tr.select_dtypes(include=[np.number])
exchange_test_num = exchange_test.select_dtypes(include=[np.number])


# In[8]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# In[11]:


import joblib

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler())
])

exchange_tr_prepared = num_pipeline.fit_transform(exchange_tr_num)
exchange_test_prepared = num_pipeline.fit_transform(exchange_test_num)

joblib.dump(num_pipeline, 'pipeline2.pkl')


# In[12]:


from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()
lin_reg.fit(exchange_tr_prepared,exchange_tr_labels)

from sklearn.tree import DecisionTreeRegressor

tree_reg=DecisionTreeRegressor()
tree_reg.fit(exchange_tr_prepared,exchange_tr_labels)

from sklearn.ensemble import RandomForestRegressor

forest_reg=RandomForestRegressor()
forest_reg.fit(exchange_tr_prepared,exchange_tr_labels)


# In[13]:


linPreds_tr=lin_reg.predict(exchange_tr_prepared)
treePreds_tr=tree_reg.predict(exchange_tr_prepared)
forestPreds_tr=forest_reg.predict(exchange_tr_prepared)

from sklearn.metrics import mean_squared_error

lin_rmse_tr=np.sqrt(mean_squared_error(exchange_tr_labels,linPreds_tr))
tree_rmse_tr=np.sqrt(mean_squared_error(exchange_tr_labels,treePreds_tr))
forest_rmse_tr=np.sqrt(mean_squared_error(exchange_tr_labels,forestPreds_tr))

print(lin_rmse_tr)
print(tree_rmse_tr)
print(forest_rmse_tr)


# In[14]:


linPreds_test=lin_reg.predict(exchange_test_prepared)
treePreds_test=tree_reg.predict(exchange_test_prepared)
forestPreds_test=forest_reg.predict(exchange_test_prepared)

from sklearn.metrics import mean_squared_error

lin_rmse_test=np.sqrt(mean_squared_error(exchange_test_labels,linPreds_test))
tree_rmse_test=np.sqrt(mean_squared_error(exchange_test_labels,treePreds_test))
forest_rmse_test=np.sqrt(mean_squared_error(exchange_test_labels,forestPreds_test))

print(lin_rmse_test)
print(tree_rmse_test)
print(forest_rmse_test)


# In[15]:


from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=1,solver="cholesky")
ridge_reg.fit(exchange_tr_prepared,exchange_tr_labels)
ridge_preds = ridge_reg.predict(exchange_test_prepared)

from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor()
sgd_reg.fit(exchange_tr_prepared,exchange_tr_labels)
sdg_preds = sgd_reg.predict(exchange_test_prepared)

from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(exchange_tr_prepared,exchange_tr_labels)
lasso_preds = lasso_reg.predict(exchange_test_prepared)

from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1,l1_ratio=0.5)
elastic_net.fit(exchange_tr_prepared,exchange_tr_labels)
elastic_preds = elastic_net.predict(exchange_test_prepared)


# In[16]:


ridge_rmse = np.sqrt(mean_squared_error(exchange_test_labels,ridge_preds))
sdg_rmse = np.sqrt(mean_squared_error(exchange_test_labels,sdg_preds))
lasso_rmse = np.sqrt(mean_squared_error(exchange_test_labels,lasso_preds))
elastic_rmse = np.sqrt(mean_squared_error(exchange_test_labels,elastic_preds))


# In[17]:


print(ridge_rmse)
print(sdg_rmse)
print(lasso_rmse)
print(elastic_rmse)


# In[18]:


joblib.dump(lin_reg, 'model2.pkl')


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[30]:


import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# In[31]:


pca = PCA()#(n_components=2)


# In[32]:


def preprocessWithAspectRatio(image,label):
    resized_image=tf.image.resize_with_pad(image,299,299)
    final_image=keras.applications.xception.preprocess_input(resized_image)
    return final_image, label


# In[33]:


base_model=keras.applications.xception.Xception(weights='imagenet',include_top=False)
base_model.summary()


# In[34]:


output=keras.layers.GlobalAveragePooling2D()(base_model.output)
#output=keras.layers.Dense(info.features['label'].num_classes,activation="softmax")(avg)
model=keras.models.Model(inputs=base_model.input,outputs=output)
model.summary()


# In[41]:


testSet, info = tfds.load(name='oxford_flowers102', split='test',as_supervised=True, with_info=True)


# In[48]:


testPipe=testSet.map(preprocessWithAspectRatio,num_parallel_calls=32).batch(1).prefetch(1)


# In[50]:


prob_predictions = model.predict(testPipe)


# In[51]:


print(len(prob_predictions))


# In[54]:


X2D = pca.fit_transform(prob_predictions)


# In[57]:


explained_var_ratio = pca.explained_variance_ratio_
print(np.shape(explained_var_ratio))


# In[60]:


sum_exp_var_ratio = np.cumsum(explained_var_ratio)
print(np.shape(sum_exp_var_ratio))


# In[67]:


plt.step(range(0, len(sum_exp_var_ratio)), sum_exp_var_ratio, where='mid')
plt.ylabel('Explained Variance')
plt.xlabel('Dimensions')
plt.title('Explained Variance Ratio vs Number of Dimensions kept under PCA')
plt.grid()
#plt.savefig('explainedVariancePlot.png')
plt.show()


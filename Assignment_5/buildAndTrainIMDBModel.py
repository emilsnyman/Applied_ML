#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import tensorflow_addons as tfa
import seaborn as sns
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from collections import Counter


# In[2]:


train_split, info = tfds.load("imdb_reviews",split='train[:6250]', as_supervised=True, with_info=True)
test_data, info = tfds.load("imdb_reviews",split='test[50%:]', as_supervised=True, with_info=True)
train_full, info = tfds.load("imdb_reviews",split='train', as_supervised=True, with_info=True)


# In[3]:


def preprocess(X_batch, y_batch):
    X_batch = tf.strings.substr(X_batch, 0, 300)
    X_batch = tf.strings.regex_replace(X_batch, rb"<br\s*/?>", b" ")
    X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z']", b" ")
    X_batch = tf.strings.split(X_batch)
    return X_batch.to_tensor(default_value=b"<pad>"), y_batch


# In[4]:


vocabulary = Counter()
for X_batch, y_batch in train_full.batch(32).map(preprocess):
    for review in X_batch:
        vocabulary.update(list(review.numpy()))
        
vocab_size = 10000
truncated_vocabulary = [
    word for word, count in vocabulary.most_common()[24:vocab_size+24]]

words = tf.constant(truncated_vocabulary)
word_ids = tf.range(len(truncated_vocabulary), dtype=tf.int64)
vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
num_oov_buckets = 1000
table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)

def encode_words(X_batch, y_batch):
    return table.lookup(X_batch), y_batch


# In[5]:


train_set = train_split.batch(32).map(preprocess)
train_set = train_set.map(encode_words).prefetch(1)
valid_data, info = tfds.load("imdb_reviews",split='test[:50%]', as_supervised=True, with_info=True)
valid_set = valid_data.batch(32).map(preprocess)
valid_set = valid_set.map(encode_words).prefetch(1)


# In[6]:


embed_size = 128
reg = tf.keras.regularizers.L2(0.001)
model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size,
                           mask_zero=True, # not shown in the book
                           input_shape=[None]),
    keras.layers.GRU(128, return_sequences=True,dropout=0.5),#,dropout=0.5),#,kernel_regularizer=reg),
    keras.layers.GRU(16),#,kernel_regularizer=reg),
    keras.layers.Dense(1, activation="sigmoid")
    ])

#model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
#history = model.fit(train_set, epochs=5)


# In[7]:


checkpoint_cb=keras.callbacks.ModelCheckpoint('IMDB_model.h5',monitor='val_loss',mode='min', save_best_only=True)

earlyStop_cb=keras.callbacks.EarlyStopping(patience=25,restore_best_weights=True,monitor='val_loss',mode='min')

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(train_set,validation_data=valid_set,epochs=2, callbacks=[checkpoint_cb,earlyStop_cb])


# In[ ]:


test_set = test_data.batch(32).prefetch(1)

results = model.evaluate(test_set)


# In[8]:


train_split, info = tfds.load("imdb_reviews",split='train[6250:12500]', as_supervised=True, with_info=True)
train_set = train_split.batch(32).map(preprocess)
train_set = train_set.map(encode_words).prefetch(1)

model.fit(train_set,validation_data=valid_set,epochs=2, callbacks=[checkpoint_cb,earlyStop_cb])


# In[ ]:


test_set = test_data.batch(32).map(preprocess)
test_set = test_set.map(encode_words).prefetch(1)

results = model.evaluate(test_set)


# In[9]:


train_split, info = tfds.load("imdb_reviews",split='train[12500:18750]', as_supervised=True, with_info=True)
train_set = train_split.batch(32).map(preprocess)
train_set = train_set.map(encode_words).prefetch(1)

model.fit(train_set,validation_data=valid_set,epochs=2, callbacks=[checkpoint_cb,earlyStop_cb])


# In[10]:


train_split, info = tfds.load("imdb_reviews",split='train[18750:]', as_supervised=True, with_info=True)
train_set = train_split.batch(32).map(preprocess)
train_set = train_set.map(encode_words).prefetch(1)

model.fit(train_set,validation_data=valid_set,epochs=2, callbacks=[checkpoint_cb,earlyStop_cb])


# In[11]:


# model.save('IMDB_sentiment_pred_best.h5')


# In[ ]:





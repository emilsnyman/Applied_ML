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


test_data, info = tfds.load("imdb_reviews",split='test', as_supervised=True, with_info=True)
train_full, info = tfds.load("imdb_reviews",split='train', as_supervised=True, with_info=True)


# In[3]:


def preprocess(X_batch, y_batch):
    X_batch = tf.strings.substr(X_batch, 0, 300)
    X_batch = tf.strings.regex_replace(X_batch, rb"<br\s*/?>", b" ")
    X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z']", b" ")
    X_batch = tf.strings.split(X_batch)
    return X_batch.to_tensor(default_value=b"<pad>"), y_batch

def encode_words(X_batch, y_batch):
    return table.lookup(X_batch), y_batch


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


# In[5]:


model = keras.models.load_model('IMDB_sentiment_pred_best.h5')


# In[52]:


def pred(reviews):
    inputs = reviews.batch(1).map(preprocess).map(encode_words).prefetch(1)
    for X_batch, y_batch in inputs:
        for review, label in zip(X_batch, y_batch.numpy()):
            y_pred = model.predict(review)
            print('Prediction: Positive Probability - ', y_pred, 'Negative Probability - ',1-y_pred)
            print("Label: ", label, "= Positive" if label else "= Negative")


# In[53]:


test = test_data.batch(1).take(5)
pred(test)


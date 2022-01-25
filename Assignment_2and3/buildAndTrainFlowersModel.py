#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from preprocessDefinition import preprocess


# In[2]:


base_model=keras.applications.xception.Xception(weights='imagenet',include_top=True)


# In[5]:


trainSet,info=tfds.load(name='oxford_flowers102', split='train+validation', as_supervised=True,with_info=True)
validSet=tfds.load(name='oxford_flowers102', split='test[90%:]',as_supervised=True)
testSet=tfds.load(name='oxford_flowers102', split='test[:90%]',as_supervised=True)


# In[6]:


nToAugment=4

def augmentImages(image,label):
    resized_image=tf.image.resize_with_pad(image,299,299)
    imageL=[resized_image]
    myGen=keras.preprocessing.image.ImageDataGenerator(rotation_range=40,
    width_shift_range=[-0.2,0.2],height_shift_range=[-.2,.2],
    brightness_range=[.6,1.0], shear_range=0.0,
    channel_shift_range=0.0, fill_mode='constant', cval=0.0, horizontal_flip=True,
    vertical_flip=True)
    augmented_images=[next(myGen.flow(resized_image)) for _ in range(nToAugment)]
    labels=[label.numpy() for _ in range(nToAugment+1)]
    imageL.extend(augmented_images)
    return imageL, labels

def augmentImagesTF(image,label):
    func=tf.py_function(augmentImages,[image,label],[tf.float32,tf.int32])
    func[0] = tf.reshape(func[0], (nToAugment+1, 299,299,3))
    func[1] = tf.reshape(func[1], (nToAugment+1, 1))
    return func

def mySqueeze(x,y):
    return tf.squeeze(x),y

trainPipeAug=trainSet.batch(1).prefetch(1).map(augmentImagesTF,num_parallel_calls=32)
trainPipeAug=trainPipeAug.unbatch().map(mySqueeze,num_parallel_calls=32).shuffle(512)


# In[7]:


def preproc(image,label):
    inp=keras.applications.xception.preprocess_input(image)
    return inp,label


# In[8]:


trainPipeAug=trainPipeAug.map(preproc,num_parallel_calls=32).batch(128).prefetch(1)
validPipe=validSet.map(preprocess,num_parallel_calls=32).batch(128).prefetch(1)


# In[9]:


base_model=keras.applications.xception.Xception(weights='imagenet',include_top=False)
base_model.summary()


# In[10]:


avg=keras.layers.GlobalAveragePooling2D()(base_model.output)
output=keras.layers.Dense(info.features['label'].num_classes,activation="softmax")(avg)
model=keras.models.Model(inputs=base_model.input,outputs=output)
model.summary()


# In[11]:


for layer in base_model.layers:
    layer.trainable=False
    
for layer in model.layers:
    print(layer.trainable)


# In[ ]:


checkpoint_cb=keras.callbacks.ModelCheckpoint('model_xCeption.h5', save_best_only=True)

earlyStop_cb=keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)

ss=1e-1

optimizer=keras.optimizers.SGD(learning_rate=ss)

model.compile(loss="sparse_categorical_crossentropy",optimizer=optimizer, metrics=["accuracy"])
model.fit(trainPipeAug,validation_data=validPipe,epochs=25, callbacks=[checkpoint_cb,earlyStop_cb])


# In[ ]:


testPipe=testSet.map(preprocessWithAspectRatio,num_parallel_calls=32).batch(32).prefetch(1)
model.evaluate(testPipe)


# In[ ]:


model.save('trained_Xception_model.h5')


# In[ ]:


#Code for second round of training, done in a new notebook, is below


# In[ ]:


trainSet,info=tfds.load(name='oxford_flowers102', split='train+validation', as_supervised=True,with_info=True)
validSet=tfds.load(name='oxford_flowers102', split='test[90%:]',as_supervised=True)
testSet=tfds.load(name='oxford_flowers102', split='test[:90%]',as_supervised=True)


# In[ ]:


trainPipe=trainSet.map(preprocess,num_parallel_calls=32).batch(128).prefetch(1)
validPipe=validSet.map(preprocess,num_parallel_calls=32).batch(128).prefetch(1)


# In[ ]:


model = keras.models.load_model('trained_Xception_model.h5')


# In[ ]:


checkpoint_cb=keras.callbacks.ModelCheckpoint('model_xCeption_2.h5', save_best_only=True)

earlyStop_cb=keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)

ss=5e-2

optimizer=keras.optimizers.SGD(learning_rate=ss)

model.compile(loss="sparse_categorical_crossentropy",optimizer=optimizer, metrics=["accuracy"])
model.fit(trainPipe,validation_data=validPipe,epochs=10, callbacks=[checkpoint_cb,earlyStop_cb])


# In[ ]:


testPipe=testSet.map(preprocessWithAspectRatio,num_parallel_calls=32).batch(32).prefetch(1)
model.evaluate(testPipe)


# In[ ]:


model.save('trained_Xception_model_2.h5')


# In[ ]:


#Code for final round of training, done in a new notebook, is below


# In[ ]:


trainSet,info=tfds.load(name='oxford_flowers102', split='train+validation',as_supervised=True,with_info=True)

validSet=tfds.load(name='oxford_flowers102', split='test[90%:]',as_supervised=True)

testSet=tfds.load(name='oxford_flowers102', split='test[:90%]',as_supervised=True)


# In[ ]:


nToAugment=4

def augmentImages(image,label):
    resized_image=tf.image.resize_with_pad(image,299,299)
    imageL=[resized_image]
    myGen=keras.preprocessing.image.ImageDataGenerator(rotation_range=40,
    width_shift_range=[-0.2,0.2],height_shift_range=[-.2,.2],
    brightness_range=[.6,1.0], shear_range=0.0,
    channel_shift_range=0.0, fill_mode='constant', cval=0.0, horizontal_flip=True,
    vertical_flip=True)
    augmented_images=[next(myGen.flow(resized_image)) for _ in range(nToAugment)]
    labels=[label.numpy() for _ in range(nToAugment+1)]
    imageL.extend(augmented_images)
    return imageL, labels

def augmentImagesTF(image,label):
    func=tf.py_function(augmentImages,[image,label],[tf.float32,tf.int32])
    func[0] = tf.reshape(func[0], (nToAugment+1, 299,299,3))
    func[1] = tf.reshape(func[1], (nToAugment+1, 1))
    return func

def mySqueeze(x,y):
    return tf.squeeze(x),y

trainPipeAug=trainSet.batch(1).prefetch(1).map(augmentImagesTF,num_parallel_calls=32)
trainPipeAug=trainPipeAug.unbatch().map(mySqueeze,num_parallel_calls=32).shuffle(512)


# In[ ]:


def preproc(image,label):
    inp=keras.applications.xception.preprocess_input(image)
    return inp,label


# In[ ]:


trainPipeAug=trainPipeAug.map(preproc,num_parallel_calls=32).batch(32).prefetch(1)
validPipe=validSet.map(preprocess,num_parallel_calls=32).batch(32).prefetch(1)
testPipe=testSet.map(preprocess,num_parallel_calls=32).batch(32).prefetch(1)


# In[ ]:


model = keras.models.load_model('trained_Xception_model_2.h5')
model.summary()


# In[ ]:


for layer in model.layers:
    layer.trainable=True
    
for layer in model.layers:
    print(layer.trainable)


# In[ ]:


checkpoint_cb=keras.callbacks.ModelCheckpoint('model_xCeption_3.h5', save_best_only=True)

earlyStop_cb=keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)

ss=2e-2

optimizer=keras.optimizers.SGD(learning_rate=ss)

model.compile(loss="sparse_categorical_crossentropy",optimizer=optimizer, metrics=["accuracy"])

model.fit(trainPipeAug,validation_data=validPipe,epochs=25, callbacks=[checkpoint_cb,earlyStop_cb])


# In[ ]:


# NOTE: Manually stopped training at epoch 16 as it was taking too long and model was no longer improving


# In[ ]:


model = keras.models.load_model('model_xCeption_3.h5')


# In[ ]:


#model.save('flowersModel.h5')


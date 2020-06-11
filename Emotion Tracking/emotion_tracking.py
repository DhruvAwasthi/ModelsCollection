#!/usr/bin/env python
# coding: utf-8

# # Emotion Detection in Images of Faces

# ## Load packages

# In[1]:


import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1 - Emotion Tracking
# **Understanding the problem statement**
# * A nearby community health clinic is helping the local residents monitor their mental health.  
# * As part of their study, they are asking volunteers to record their emotions throughout the day.
# * To help the participants more easily track their emotions, we are asked to create an app that will classify their emotions based on some pictures that the volunteers will take of their facial expressions.
# * As a proof-of-concept, we will first train our model to detect if someone's emotion is classified as "happy" or "not happy."
# 
# To build and train this model, we have gathered pictures of some volunteers in a nearby neighborhood. The dataset is labeled.
# <img src="images/face_images.png" style="width:550px;height:250px;">
# 
# As a first step let's normalize the dataset and learn about its shapes.

# In[2]:


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))


# **Details of the "Face" dataset**:
# - Images are of shape (64,64,3)
# - Training: 600 pictures
# - Test: 150 pictures

# ## 2 - Building a model 

# In[22]:


def HappyModel(input_shape):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset
        (height, width, channels) as a tuple.  
        Note that this does not include the 'batch' as a dimension.
        If you have a batch like 'X_train', 
        then you can provide the input_shape using
        X_train.shape[1:]
    """
    """
    Returns:
    model -- a Model() instance in Keras
    """

    X_input = Input(input_shape)
    X = ZeroPadding2D((3, 3))(X_input)
    X = Conv2D(32, (3, 3), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool')(X)
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)
    model = Model(inputs = X_input, outputs = X, name='HappyModel')
    
    return model


# #### Step 1: create the model.  

# In[23]:


happyModel = HappyModel(X_train.shape[1:])


# #### Step 2: compile the model

# In[24]:


happyModel.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# #### Step 3: train the model

# In[25]:


happyModel.fit(X_train, Y_train, epochs=40, batch_size=32)


# #### Step 4: evaluate model  

# In[26]:


preds = happyModel.evaluate(X_test, Y_test, batch_size=32)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


# ## 4 - Test with your friend's image 

# In[27]:


img_path = 'images/my_image.jpg'
img = image.load_img(img_path, target_size=(64, 64))
imshow(img)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

print(happyModel.predict(x))


# In[28]:


happyModel.summary()


# In[29]:


plot_model(happyModel, to_file='HappyModel.png')
SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))


#!/usr/bin/env python
# coding: utf-8

# Below is code with a link to a happy or sad dataset which contains 80 images, 40 happy and 40 sad. 
# Create a convolutional neural network that trains to 100% accuracy on these images,  which cancels training upon hitting training accuracy of >.999
# 
# Hint -- it will work best with 3 convolutional layers.

# In[1]:


import tensorflow as tf
import os
import zipfile
from os import path, getcwd, chdir

get_ipython().system('wget --no-check-certificate     "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/happy-or-sad.zip"     -O "/tmp/happy-or-sad.zip"')

zip_ref = zipfile.ZipFile("/tmp/happy-or-sad.zip", 'r')
zip_ref.extractall("/tmp/h-or-s")
zip_ref.close()


# In[18]:


def train_happy_sad_model():
    DESIRED_ACCURACY = 0.999

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, log={}):
            if (log.get('acc')>DESIRED_ACCURACY):
                print("\nAccuracy achieved {:.1f}% so cancelling training.".format(DESIRED_ACCURACY))
                self.model.stop_training = True

    callbacks = myCallback()
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(4, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(4, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.summary()
    
    from tensorflow.keras.optimizers import RMSprop

    model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1./255)

    # Please use a target_size of 150 X 150.
    train_generator = train_datagen.flow_from_directory(
        '/tmp/h-or-s',
        target_size=(150, 150),
        batch_size=128,
        class_mode='binary')

    history = model.fit_generator(train_generator, steps_per_epoch=8, epochs=15, verbose=1, callbacks=[callbacks])
    # model fitting
    return history.history['acc'][-1]


# In[19]:


train_happy_sad_model()


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt
import sys

from random import shuffle
from skimage.io import imread
from skimage.transform import resize
from skimage.io import imread
from skimage.transform import rescale
from skimage.transform import rotate
from skimage import exposure
import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization,SpatialDropout2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img


# In[2]:


def Alex_model_binary(img_ch, img_width, img_height,base_dense,dropout=False,dr=0.2,Batch_normalization=False,spatial_dropout=False,spdr=0.1):
    
    if Batch_normalization:
        
        model = Sequential()
        model.add(Conv2D(filters=base_dense, input_shape=(img_width, img_height, img_ch),
        kernel_size=(3,3), strides=(1,1), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(filters= base_dense *2, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(filters= base_dense *4, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *4, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(filters= base_dense *2, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Flatten()) 
    
    else :
        if spatial_dropout:
            
            model = Sequential()
            model.add(Conv2D(filters=base_dense, input_shape=(img_width, img_height, img_ch),
            kernel_size=(3,3), strides=(1,1), padding='same')) 
            model.add(Activation('relu'))
            model.add(SpatialDropout2D(spdr))
            model.add(MaxPooling2D(pool_size=(2,2)))

            model.add(Conv2D(filters= base_dense *2, kernel_size=(3,3), strides=(1,1), padding='same')) 
            model.add(Activation('relu'))
            model.add(SpatialDropout2D(spdr))
            model.add(MaxPooling2D(pool_size=(2,2)))

            model.add(Conv2D(filters= base_dense *4, kernel_size=(3,3), strides=(1,1), padding='same')) 
            model.add(Activation('relu'))
            model.add(SpatialDropout2D(spdr))
            model.add(Conv2D(filters= base_dense *4, kernel_size=(3,3), strides=(1,1), padding='same')) 
            model.add(Activation('relu'))
            model.add(SpatialDropout2D(spdr))

            model.add(Conv2D(filters= base_dense *2, kernel_size=(3,3), strides=(1,1), padding='same')) 
            model.add(Activation('relu'))
            model.add(SpatialDropout2D(spdr))
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(Flatten()) 
        else :
            model = Sequential()
            model.add(Conv2D(filters=base_dense, input_shape=(img_width, img_height, img_ch),
            kernel_size=(3,3), strides=(1,1), padding='same')) 
            model.add(Activation('relu'))
            
            model.add(MaxPooling2D(pool_size=(2,2)))

            model.add(Conv2D(filters= base_dense *2, kernel_size=(3,3), strides=(1,1), padding='same')) 
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))

            model.add(Conv2D(filters= base_dense *4, kernel_size=(3,3), strides=(1,1), padding='same')) 
            model.add(Activation('relu'))
            model.add(Conv2D(filters= base_dense *4, kernel_size=(3,3), strides=(1,1), padding='same')) 
            model.add(Activation('relu'))

            model.add(Conv2D(filters= base_dense *2, kernel_size=(3,3), strides=(1,1), padding='same')) 
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(Flatten()) 
    
    if dropout:
        
        model.add(Dense(64))
        model.add(Dropout(dr))
        model.add(Activation('relu'))
    
        model.add(Dense(64)) 
        model.add(Dropout(dr))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Dropout(dr))
    else:
        model.add(Dense(64))
        model.add(Activation('relu'))
    
        model.add(Dense(64)) 
        model.add(Activation('relu'))
        model.add(Dense(1))
        
    model.add(Activation('sigmoid'))
    model.summary() 
    return model


# In[3]:


def vg_model(img_ch, img_width, img_height,base_dense,dropout=False,dr=0.2,Batch_normalization=False):
    if Batch_normalization:
        
        #blue layer
        model = Sequential()
        model.add(Conv2D(filters=base_dense, input_shape=(img_width, img_height, img_ch),
        kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *2, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))


        #orange layer
        model.add(Conv2D(filters= base_dense *2, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *2, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *2, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        #purple layer
        model.add(Conv2D(filters= base_dense *4, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *4, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *4, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        #green layer
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        #red layer
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
    else:
        #blue layer
        model = Sequential()
        model.add(Conv2D(filters=base_dense, input_shape=(img_width, img_height, img_ch),
        kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *2, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))


        #orange layer
        model.add(Conv2D(filters= base_dense *2, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *2, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *2, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        #purple layer
        model.add(Conv2D(filters= base_dense *4, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *4, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *4, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        #green layer
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        #red layer
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
    if dropout:
        #dense layer
        model.add(Flatten()) 
        model.add(Dense(64))
        model.add(Dropout(dr))
        model.add(Activation('relu'))

        model.add(Dense(64)) 
        model.add(Dropout(dr))
        model.add(Activation('relu'))

        model.add(Dense(64))
        model.add(Dropout(dr))
        model.add(Activation('relu'))

        model.add(Dense(1))
        model.add(Dropout(dr))
        model.add(Activation('sigmoid'))
    else:
        #dense layer
        model.add(Flatten()) 
        model.add(Dense(64))
        model.add(Activation('relu'))

        model.add(Dense(64)) 
        model.add(Activation('relu'))

        model.add(Dense(64))
        model.add(Activation('relu'))

        model.add(Dense(1))
        model.add(Activation('sigmoid'))
    
    
    model.summary() 
    return model


# In[4]:


def MLP(dr,train_data):
    clf = Sequential()
    clf.add(Dense(128,activation='relu',input_shape = train_data.shape[1:]))
    clf.add(Flatten())    
    clf.add(Dropout(dr))
    clf.add(Dense(1,activation='sigmoid'))
    
    clf.summary()

    return clf


# In[5]:


def vg_model_multi_classes(img_ch, img_width, img_height,base_dense,dropout=False,dr=0.2,Batch_normalization=False):
    if Batch_normalization:
        
        #blue layer
        model = Sequential()
        model.add(Conv2D(filters=base_dense, input_shape=(img_width, img_height, img_ch),
        kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *2, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))


        #orange layer
        model.add(Conv2D(filters= base_dense *2, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *2, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *2, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        #purple layer
        model.add(Conv2D(filters= base_dense *4, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *4, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *4, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        #green layer
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        #red layer
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
    else:
        #blue layer
        model = Sequential()
        model.add(Conv2D(filters=base_dense, input_shape=(img_width, img_height, img_ch),
        kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *2, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))


        #orange layer
        model.add(Conv2D(filters= base_dense *2, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *2, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *2, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        #purple layer
        model.add(Conv2D(filters= base_dense *4, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *4, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *4, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        #green layer
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        #red layer
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same')) 
        model.add(Activation('relu'))
        model.add(Conv2D(filters= base_dense *8, kernel_size=(3,3), strides=(1,1), padding='same',name = 'Last_ConvLayer')) 
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
    if dropout:
        #dense layer
        model.add(Flatten()) 
        model.add(Dense(64))
        model.add(Dropout(dr))
        model.add(Activation('relu'))

        model.add(Dense(64)) 
        model.add(Dropout(dr))
        model.add(Activation('relu'))

        model.add(Dense(64))
        model.add(Dropout(dr))
        model.add(Activation('relu'))

        model.add(Dense(2))
        model.add(Dropout(dr))
        model.add(Activation('softmax'))
    else:
        #dense layer
        model.add(Flatten()) 
        model.add(Dense(64))
        model.add(Activation('relu'))

        model.add(Dense(64)) 
        model.add(Activation('relu'))

        model.add(Dense(64))
        model.add(Activation('relu'))

        model.add(Dense(2))
        model.add(Activation('softmax'))
    
    
    model.summary() 
    return model


# In[6]:


def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))
    model.summary()


# In[ ]:





# In[ ]:





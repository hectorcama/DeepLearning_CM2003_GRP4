#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import random 
import re
import cv2

from random import shuffle
from skimage.io import imread
from skimage.transform import resize

from skimage.transform import rescale
from skimage.transform import rotate
from skimage import exposure
import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization,SpatialDropout2D,Conv2DTranspose,concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img


# In[2]:


_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]


# In[2]:


def create_set_dir(lab_file,type_file,set_file):
    data_path='../DL_course_data/'
    set_dir=os.path.join(data_path, lab_file,type_file,set_file)
    return set_dir


# In[3]:


def create_set_list(set_dir):
    set_list = os.listdir(set_dir)
    set_list.sort(key=natural_sort_key)
    return set_list


# In[4]:


def list_dir(data_list,data_dir):
    new_data=[]
    for x in data_list:
        new = os.path.join(data_dir,x)
        new_data.append(new)
    return new_data   


# In[ ]:


def create_data(lab_file,type_file,train_percent):
    img_dir1 = create_set_dir(lab_file,type_file,'Image')
    img_dir2 = create_set_dir(lab_file,type_file,'Mask')

    image_list = create_set_list(img_dir1)
    mask_list = create_set_list(img_dir2)

    index_position = list(zip(image_list,mask_list))
    random.shuffle(index_position)
    image_list[:],mask_list[:] = zip(*index_position)

    length = len(image_list)
    train_length = int(length*train_percent)

    x_train = image_list[0:train_length]
    y_train = mask_list[0:train_length]
    x_test = image_list[train_length:]
    y_test = mask_list[train_length:]
    
    x_train = list_dir(x_train,img_dir1)
    y_train = list_dir(y_train,img_dir2)
    x_test = list_dir(x_test,img_dir1)
    y_test = list_dir(y_test,img_dir2)
    return x_train,y_train,x_test,y_test


# In[ ]:


def load_data(data_list,img_w,img_h,img_ch,mask=False):
    tab = np.zeros((len(data_list),img_w,img_h,img_ch),dtype='float32')
    for i in range(len(data_list)):
        Img = cv2.imread(data_list[i],0)
        Img = cv2.resize(Img, (img_w, img_h))
        Img = Img.reshape(img_w,img_h)/255
        if mask:
            Img[Img>0]=1
            Img[Img!=1]=0
        tab[i,:,:,0]=Img
    return tab


# In[7]:


def augmentation(image_set,mask_set,dictionary_augmentation,batch_size):
    
    image_datagen = ImageDataGenerator(**dictionary_augmentation)
    mask_datagen = ImageDataGenerator(**dictionary_augmentation)

    image_generator = image_datagen.flow(
    image_set,
    y=None,
    batch_size=batch_size,
    shuffle=False,
    seed=1)
    
    mask_generator = mask_datagen.flow(
    mask_set,
    y=None,
    batch_size=batch_size,
    shuffle=False,
    seed=1)
    train_generator=(pair for pair in zip(image_generator,mask_generator))
    
    
    STEP_SIZE_TRAIN=(len(image_set)+len(mask_set))//batch_size
    
    return train_generator,STEP_SIZE_TRAIN


# In[ ]:


def conv_block(InputLayer,base_dense,BatchNorm=False):
    if BatchNorm ==True:
        conv1 = Conv2D(base_dense,(3,3),strides = (1,1), padding = "same")(InputLayer)
        batch1 = BatchNormalization()(conv1)
        act1 = Activation('relu')(batch1)
        conv2 = Conv2D(base_dense,(3,3),strides = (1,1), padding = "same")(act1)
        batch2 = BatchNormalization()(conv2)
        act2 = Activation('relu')(batch2)

    else:
        conv1 = Conv2D(base_dense,(3,3),strides = (1,1), padding = "same", activation="relu")(InputLayer)
        conv2 = Conv2D(base_dense,(3,3),strides = (1,1), padding = "same", activation="relu")(conv1)
    return conv2


# In[ ]:


def get_unet(base_dense,img_w,img_h,img_ch,dropout=False,dr=0.2):
    input_size = (img_w, img_h,img_ch)
    input_layer = Input(shape=input_size, name='input_layer')
    
    if dropout==True:
        conv1 = conv_block(input_layer,base_dense,BatchNorm=False)
        pool1 = MaxPooling2D((2,2))(conv1)
        pool1 = Dropout(dr)(pool1)

        conv2 = conv_block(pool1,base_dense*2,BatchNorm=False)
        pool2 = MaxPooling2D((2,2))(conv2)
        pool2 = Dropout(dr)(pool2)

        conv3 = conv_block(pool2,base_dense*4,BatchNorm=False)
        pool3 = MaxPooling2D((2,2))(conv3)
        pool3 = Dropout(dr)(pool3)

        conv4 = conv_block(pool3,base_dense*8,BatchNorm=False)
        pool4 = MaxPooling2D((2,2))(conv4)
        pool4 = Dropout(dr)(pool4)

        #middle
        convm = conv_block(pool4,base_dense*16,BatchNorm=False)

        #deconvolution
        deconv1 = Conv2DTranspose(base_dense*8,(3,3),strides=(2,2),padding="same",activation="relu")(convm)
        uconv1=concatenate([deconv1,conv4])
        uconv1=Dropout(dr)(uconv1)
        uconv1=conv_block(uconv1,base_dense*8,BatchNorm=False)

        deconv2 = Conv2DTranspose(base_dense*4,(3,3),strides=(2,2),padding="same",activation="relu")(uconv1)
        uconv2=concatenate([deconv2,conv3])
        uconv2=Dropout(dr)(uconv2)
        uconv2=conv_block(uconv2,base_dense*4,BatchNorm=False)

        deconv3= Conv2DTranspose(base_dense*2,(3,3),strides=(2,2),padding="same",activation="relu")(uconv2)
        uconv3=concatenate([deconv3,conv2])
        uconv3=Dropout(dr)(uconv3)
        uconv3=conv_block(uconv3,base_dense*2,BatchNorm=False)

        deconv4 = Conv2DTranspose(base_dense,(3,3),strides=(2,2),padding="same",activation="relu")(uconv3)
        uconv4=concatenate([deconv4,conv1])
        uconv4=Dropout(dr)(uconv4)
        uconv4=conv_block(uconv4,base_dense,BatchNorm=False)
    
    
    else:
        conv1 = conv_block(input_layer,base_dense,BatchNorm=False)
        pool1 = MaxPooling2D((2,2))(conv1)

        conv2 = conv_block(pool1,base_dense*2,BatchNorm=False)
        pool2 = MaxPooling2D((2,2))(conv2)

        conv3 = conv_block(pool2,base_dense*4,BatchNorm=False)
        pool3 = MaxPooling2D((2,2))(conv3)

        conv4 = conv_block(pool3,base_dense*8,BatchNorm=False)
        pool4 = MaxPooling2D((2,2))(conv4)

        #middle
        convm = conv_block(pool4,base_dense*16,BatchNorm=False)

        #deconvolution
        deconv1 = Conv2DTranspose(base_dense*8,(3,3),strides=(2,2),padding="same",activation="relu")(convm)
        uconv1=concatenate([deconv1,conv4])
        uconv1=conv_block(uconv1,base_dense*8,BatchNorm=False)

        deconv2 = Conv2DTranspose(base_dense*4,(3,3),strides=(2,2),padding="same",activation="relu")(uconv1)
        uconv2=concatenate([deconv2,conv3])
        uconv2=conv_block(uconv2,base_dense*4,BatchNorm=False)

        deconv3= Conv2DTranspose(base_dense*2,(3,3),strides=(2,2),padding="same",activation="relu")(uconv2)
        uconv3=concatenate([deconv3,conv2])
        uconv3=conv_block(uconv3,base_dense*2,BatchNorm=False)

        deconv4 = Conv2DTranspose(base_dense,(3,3),strides=(2,2),padding="same",activation="relu")(uconv3)
        uconv4=concatenate([deconv4,conv1])
        uconv4=conv_block(uconv4,base_dense,BatchNorm=False)
        
    output_layer=Conv2D(1,(1,1),padding='same',activation='sigmoid',name='output_layer')(uconv4)
    
    model=Model(inputs=input_layer, outputs=output_layer)
    model.summary()
    
    return model


# In[11]:


def get_unet_batch(base_dense,img_w,img_h,img_ch,dropout=False,dr=0.2):
    input_size = (img_w, img_h,img_ch)
    input_layer = Input(shape=input_size, name='input_layer')
    
    if dropout==True:
        conv1 = conv_block(input_layer,base_dense,BatchNorm=True)
        pool1 = MaxPooling2D((2,2))(conv1)
        pool1 = Dropout(dr)(pool1)

        conv2 = conv_block(pool1,base_dense*2,BatchNorm=True)
        pool2 = MaxPooling2D((2,2))(conv2)
        pool2 = Dropout(dr)(pool2)

        conv3 = conv_block(pool2,base_dense*4,BatchNorm=True)
        pool3 = MaxPooling2D((2,2))(conv3)
        pool3 = Dropout(dr)(pool3)

        conv4 = conv_block(pool3,base_dense*8,BatchNorm=True)
        pool4 = MaxPooling2D((2,2))(conv4)
        pool4 = Dropout(dr)(pool4)

        #middle
        convm = conv_block(pool4,base_dense*16,BatchNorm=True)

        #deconvolution
        deconv1 = Conv2DTranspose(base_dense*8,(3,3),strides=(2,2),padding="same",activation="relu")(convm)
        uconv1=concatenate([deconv1,conv4])
        uconv1=Dropout(dr)(uconv1)
        uconv1=conv_block(uconv1,base_dense*8,BatchNorm=True)

        deconv2 = Conv2DTranspose(base_dense*4,(3,3),strides=(2,2),padding="same",activation="relu")(uconv1)
        uconv2=concatenate([deconv2,conv3])
        uconv2=Dropout(dr)(uconv2)
        uconv2=conv_block(uconv2,base_dense*4,BatchNorm=True)

        deconv3= Conv2DTranspose(base_dense*2,(3,3),strides=(2,2),padding="same",activation="relu")(uconv2)
        uconv3=concatenate([deconv3,conv2])
        uconv3=Dropout(dr)(uconv3)
        uconv3=conv_block(uconv3,base_dense*2,BatchNorm=True)

        deconv4 = Conv2DTranspose(base_dense,(3,3),strides=(2,2),padding="same",activation="relu")(uconv3)
        uconv4=concatenate([deconv4,conv1])
        uconv4=Dropout(dr)(uconv4)
        uconv4=conv_block(uconv4,base_dense,BatchNorm=True)
    
    
    else:
        conv1 = conv_block(input_layer,base_dense,BatchNorm=True)
        pool1 = MaxPooling2D((2,2))(conv1)

        conv2 = conv_block(pool1,base_dense*2,BatchNorm=True)
        pool2 = MaxPooling2D((2,2))(conv2)

        conv3 = conv_block(pool2,base_dense*4,BatchNorm=True)
        pool3 = MaxPooling2D((2,2))(conv3)

        conv4 = conv_block(pool3,base_dense*8,BatchNorm=True)
        pool4 = MaxPooling2D((2,2))(conv4)

        #middle
        convm = conv_block(pool4,base_dense*16,BatchNorm=True)

        #deconvolution
        deconv1 = Conv2DTranspose(base_dense*8,(3,3),strides=(2,2),padding="same",activation="relu")(convm)
        uconv1=concatenate([deconv1,conv4])
        uconv1=conv_block(uconv1,base_dense*8,BatchNorm=True)

        deconv2 = Conv2DTranspose(base_dense*4,(3,3),strides=(2,2),padding="same",activation="relu")(uconv1)
        uconv2=concatenate([deconv2,conv3])
        uconv2=conv_block(uconv2,base_dense*4,BatchNorm=True)

        deconv3= Conv2DTranspose(base_dense*2,(3,3),strides=(2,2),padding="same",activation="relu")(uconv2)
        uconv3=concatenate([deconv3,conv2])
        uconv3=conv_block(uconv3,base_dense*2,BatchNorm=True)

        deconv4 = Conv2DTranspose(base_dense,(3,3),strides=(2,2),padding="same",activation="relu")(uconv3)
        uconv4=concatenate([deconv4,conv1])
        uconv4=conv_block(uconv4,base_dense,BatchNorm=True)
        
    output_layer=Conv2D(1,(1,1),padding='same',activation='sigmoid',name='output_layer')(uconv4)
    
    model=Model(inputs=input_layer, outputs=output_layer)
    model.summary()
    
    return model


# In[12]:


def compile_fit_generator(model,train_generator,x_test,y_test,loss_function, optimizer,metrics,batch_size,n_epochs,STEP_SIZE_TRAIN):
    
    model.compile(loss=loss_function,optimizer = optimizer,metrics=[metrics])


    
  
    model_hist=model.fit_generator(generator=train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=(x_test,y_test)
                        ,epochs=n_epochs,verbose=1)
    return model_hist


# In[13]:


def compile_fit(model,loss_function, optimizer,metrics,x_train,y_train,x_test,y_test,batch_size,n_epochs):
    clf=model
    clf.compile(loss=loss_function,optimizer = optimizer,metrics=[metrics])
    clf_hist=clf.fit(x_train,y_train,batch_size,n_epochs,validation_data=(x_test, y_test))
    return clf_hist


# In[14]:


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


# In[15]:


def loss_curves_plot(model_hist):
    get_ipython().run_line_magic('matplotlib', 'inline')

    plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(model_hist.history["loss"], label="loss")
    plt.plot(model_hist.history["val_loss"], label="val_loss")
    plt.plot( np.argmin(model_hist.history["val_loss"]),
     np.min(model_hist.history["val_loss"]),
     marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend();
    
def accuracy_curves_plot(model_hist,metrics):
    get_ipython().run_line_magic('matplotlib', 'inline')
    
    plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(model_hist.history[metrics], label="accuracy")
    plt.plot(model_hist.history["val_"+metrics], label="val_accuracy")
    plt.plot( np.argmax(model_hist.history["val_"+metrics]),
     np.max(model_hist.history["val_"+metrics]),
     marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy Value")
    plt.legend();


# In[ ]:





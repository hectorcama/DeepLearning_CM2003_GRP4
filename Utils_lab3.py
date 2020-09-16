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
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization,SpatialDropout2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img


# In[2]:


#This function takes the image name and the two pattern names to test and return 0 if pat1 is in im_name and 1 
#if pat2 is in im_name
def gen_labels(im_name, pat1, pat2):
 
    if pat1 in im_name:
         label = np.array([0])
     
    elif pat2 in im_name:
         label = np.array([1])
     
    return label


#the function creates a list which associates each image with its corresponding label
def get_data(data_path, data_list, img_h, img_w):
     
    img_labels = []
    
    for item in enumerate(data_list):
        img = imread(os.path.join(data_path, item[1]), as_gray = True) # "as_grey"
        #each image is resized so as all image have the same dimension for the process
        img = resize(img, (img_h, img_w), anti_aliasing = True).astype('float32')
        img_labels.append([np.array(img), gen_labels(item[1], 'Mel', 'Nev')])

        if item[0] % 100 == 0:
            print('Reading: {0}/{1} of train images'.format(item[0], len(data_list)))

    shuffle(img_labels)
    return img_labels


#separates the list img_labels obtained with the function get_data into two lists
#one list with the images and one list with the labels
def get_data_arrays(nested_list, img_h, img_w):
    img_arrays = np.zeros((len(nested_list), img_h, img_w), dtype = np.float32)
    label_arrays = np.zeros((len(nested_list)), dtype = np.int32)
    for ind in range(len(nested_list)):
        img_arrays[ind] = nested_list[ind][0]
        label_arrays[ind] = nested_list[ind][1]
    img_arrays = np.expand_dims(img_arrays, axis =3)
    return img_arrays, label_arrays


#this function uses all the previous functions to create a train set and a test set 
def get_train_test_arrays(train_data_path, test_data_path, train_list,test_list, img_h, img_w):
    train_data = get_data(train_data_path, train_list, img_h, img_w)
    test_data = get_data(test_data_path, test_list, img_h, img_w)

    train_img, train_label = get_data_arrays(train_data, img_h, img_w)
    test_img, test_label = get_data_arrays(test_data, img_h, img_w)
    del(train_data)
    del(test_data)
    return train_img, test_img, train_label, test_label


# In[3]:


def create_set_dir(lab_file,type_file,set_file):
    data_path='../DL_course_data/'
    set_dir=os.path.join(data_path, lab_file,type_file,set_file)
    return set_dir
    


# In[4]:


def create_set_list(set_dir):
    set_list = os.listdir(set_dir)
    return set_list


# In[5]:


def load_data(img_h,img_w,lab_file,type_file,set_file_train,set_file_test):
    
    
    
    train_data_path = create_set_dir(lab_file,type_file,set_file_train)
    test_data_path = create_set_dir(lab_file,type_file,set_file_test)
    train_list = create_set_list(train_data_path)
    test_list = create_set_list(test_data_path)

    #creating the sets
    x_train, x_test, y_train, y_test = get_train_test_arrays(train_data_path, test_data_path,train_list, test_list, img_h, img_w)
    
    return x_train, x_test, y_train, y_test


# In[6]:


#task6 to get the pattern length
def get_length(Path, Pattern):
 # Pattern: name of the subdirectory
    Length = len(os.listdir(os.path.join(Path, Pattern)))
    return Length


# In[7]:


# number of data for each class
#name is either Bone or Skin

def len_set(name_set,train_data_dir,validation_data_dir):
    if name_set=='Skin':
        
        Len_C1_Train = get_length(train_data_dir,'Mel')
        Len_C2_Train = get_length(train_data_dir,'Nevi')
        Len_C1_Val = get_length(validation_data_dir,'Mel')
        Len_C2_Val = get_length(validation_data_dir,'Nevi')
    
    elif name_set=='Bone':
        
        Len_C1_Train = get_length(train_data_dir,'AFF')
        Len_C2_Train = get_length(train_data_dir,'NFF')
        Len_C1_Val = get_length(validation_data_dir,'AFF')
        Len_C2_Val = get_length(validation_data_dir,'NFF')
        
    return Len_C1_Train,Len_C2_Train,Len_C1_Val,Len_C2_Val


# In[8]:


def compute_fit(model,loss_function, optimizer,metrics,x_train,y_train,x_test,y_test,batch_size,n_epochs):
    clf=model
    clf.compile(loss=loss_function,optimizer = optimizer,metrics=[metrics])
    clf_hist=clf.fit(x_train,y_train,batch_size,n_epochs,validation_data=(x_test, y_test))
    return clf_hist


# In[9]:


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


# In[10]:


#metrics is a str
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


# In[11]:


def compile_fit_generator(model,train_generator,val_generator,loss_function, optimizer,metrics,batch_size,n_epochs):
    
    model.compile(loss=loss_function,optimizer = optimizer,metrics=[metrics])


    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=val_generator.n//val_generator.batch_size
    model_hist=model.fit_generator(generator=train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=val_generator,
                        validation_steps=STEP_SIZE_VALID,epochs=n_epochs,verbose=1)
    return model_hist


# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt

from random import shuffle
from skimage.io import imread
from skimage.transform import resize

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D


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
        img_labels.append([np.array(img), gen_labels(item[1], 'AFF', 'NFF')])

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


# In[ ]:





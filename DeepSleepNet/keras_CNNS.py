# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 00:37:15 2019

@author: 姜海洋
"""

import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization 
from keras.layers import Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model, Sequential
from keras.initializers import glorot_uniform
import tensorflow as tf

def double_CNNs(X, name):
 
    conv_name = 'conv' +  name
    bn_name = 'bn' +  name
    Fs = 100
    #print(X)
    X1 = Conv2D(filters = 64, kernel_size = (50,1), strides = (int(Fs/16),1), 
               padding = 'same', name = conv_name + '2a', 
               kernel_initializer = glorot_uniform(seed = 0))(X)
    X1 = BatchNormalization(axis = 3, name = bn_name + '2a')(X1)
    X1 = Activation('relu')(X1)
    #print(X1)
    X1 = MaxPooling2D([8, 1], [8,1])(X1)
    #print(X1)

    X1 = Conv2D(filters = 128, kernel_size = (8,8), strides = (1,1), 
               padding = 'same', name = conv_name + '2b', 
               kernel_initializer = glorot_uniform(seed = 0))(X1)
    X1 = BatchNormalization(axis = 3, name = bn_name + '2b')(X1)
    X1 = Activation('relu')(X1)
    #print(X1)
 
    X1 = Conv2D(filters = 128, kernel_size = (8,8), strides = (1,1), 
               padding = 'same', name = conv_name + '2c', 
               kernel_initializer = glorot_uniform(seed = 0))(X1)
    X1 = BatchNormalization(axis = 3, name = bn_name + '2c')(X1)
    print(X1)
    
    X1 = Conv2D(filters = 128, kernel_size = (8,8), strides = (1,1), 
               padding = 'same', name = conv_name + '2dc', 
               kernel_initializer = glorot_uniform(seed = 0))(X1)
    X1 = BatchNormalization(axis = 3, name = bn_name + '2d')(X1)
    X1 = MaxPooling2D([4, 1], [4,1])(X1)
    #print(X1)
###########################################################################
    X2 = Conv2D(filters = 64, kernel_size = (Fs*4,1), strides = (int(Fs/2),1), 
               padding = 'same', name = conv_name + '2a', 
               kernel_initializer = glorot_uniform(seed = 0))(X)
    X2 = BatchNormalization(axis = 3, name = bn_name + '2a')(X2)
    X2 = Activation('relu')(X2)
    #print(X2)
    X2 = MaxPooling2D([4, 1], [4,1])(X2)
    #print(X2)

    X2 = Conv2D(filters = 128, kernel_size = (6,6), strides = (1,1), 
               padding = 'same', name = conv_name + '2b', 
               kernel_initializer = glorot_uniform(seed = 0))(X2)
    X2 = BatchNormalization(axis = 3, name = bn_name + '2b')(X2)
    X2 = Activation('relu')(X2)
    #print(X2)
 
    X2 = Conv2D(filters = 128, kernel_size = (6,6), strides = (1,1), 
               padding = 'same', name = conv_name + '2c', 
               kernel_initializer = glorot_uniform(seed = 0))(X2)
    X2 = BatchNormalization(axis = 3, name = bn_name + '2c')(X2)
    #print(X2)
    
    X2 = Conv2D(filters = 128, kernel_size = (6,6), strides = (1,1), 
               padding = 'same', name = conv_name + '2dc', 
               kernel_initializer = glorot_uniform(seed = 0))(X2)
    X2 = BatchNormalization(axis = 3, name = bn_name + '2d')(X2)
    X2 = MaxPooling2D([2, 1], [2,1])(X2)
    #print(X2)
    
    y_drop_ = tf.concat([X1,X2], 1)
    y_drop = tf.nn.dropout(y_drop_, keep_prob=0.5, name='drop')
    drop_shape = y_drop.get_shape().as_list() 
    nodes = drop_shape[1] * drop_shape[2] * drop_shape[3] 
    #print('VVVVVVVVV', drop_shape)
    reshaped = tf.reshape(y_drop, [drop_shape[0], nodes],name="reshaped2fc") 

    return reshaped
# =============================================================================
# 
# X = tf.placeholder(tf.float32, [64, 3000, 1, 1])
# 
# Y = double_CNNs(X, 'LL')
# print(Y.get_shape)
# 
# 
# =============================================================================







# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 13:22:03 2019

@author: 姜海洋
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.layers.core import Dense
from keras_CNNS import double_CNNs

hidden_size = 512
Fs = 100
seq_length = 30
frame_size = 100
n_classes = 5

def relu(x):
    return tf.nn.relu(x)

def norm(input_data, name):
    with tf.variable_scope(name):
        output = tf.contrib.layers.batch_norm(input_data)
    return output

def relu_norm(input_img, name):
    out = norm(input_img, name) 
    out = relu(out) 
    return out 

def conv_1d(name, input_var, filter_shape, stride, padding="SAME", 
            bias=None, wd=None):
    with tf.variable_scope(name):
        stddev = np.sqrt(2.0/np.prod(filter_shape))
        initializer = tf.truncated_normal_initializer(stddev=stddev)   
        var = tf.get_variable("weights", filter_shape, initializer=initializer)
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var),wd,name="weight_loss")
            tf.add_to_collection("losses", weight_decay)
        kernel = var
#        print(kernel)
#        print(stride)
        output_var = tf.nn.conv2d(
            input_var,
            kernel,
            [1, stride, 1, 1],
            padding=padding
        )
        
        output_var = relu_norm(output_var, 'bnr')

        return output_var

def max_pool_1d(name, input_var, pool_size, stride, padding="SAME"):
    output_var = tf.nn.max_pool(
        input_var,
        ksize=[1, pool_size, 1, 1],
        strides=[1, stride, 1, 1],
        padding=padding,
        name=name
    )

    return output_var

def fc(name, input_var, n_hiddens, bias=None, wd=None):
    with tf.variable_scope(name):
        # Get input dimension
        input_dim = input_var.get_shape()[-1].value
        shape=[input_dim, n_hiddens]
        stddev = np.sqrt(2.0/np.prod([shape]))
        initializer = tf.truncated_normal_initializer(stddev=stddev)   
        var = tf.get_variable("weights", shape, initializer=initializer)
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var),wd,name="weight_loss")
            tf.add_to_collection("losses", weight_decay)
        weights = var
        # Multiply weights
        output_var = tf.matmul(input_var, weights)
        output_var = relu_norm(output_var, 'bnr')
        
        return output_var
        
def lstm_(x, name):
    with tf.variable_scope(name):
        X = []
        #x = tf.reshape(x,shape=[-1,x.get_shape()[-1]], name="3")
        print('before split:', x)
        x = tf.split(x,x.get_shape()[-1], -1, name="4")
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n ', type(x))
        print('ASDFGHJKLQWERTYUI', len(x), x[0])
        for i in range(len(x)):
            X.append(tf.reshape(x[i], [64, x[i].get_shape()[1]]))
        
        lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size) # 正向RNN,输出神经元数量为128
     
        lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size) # 反向RNN,输出神经元数量为128
     
        output, fw_state, bw_state = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, X, dtype=tf.float32)
        print('QQQ', len(output))
        output = tf.transpose(output,[1,2,0], name="2")
        
    return output

def ann(name, input_data):
    with tf.variable_scope(name):
        weights=tf.get_variable("weights", [input_data.get_shape()[1],n_classes])
        bias=tf.get_variable("bias",[n_classes])  
        out = tf.matmul(input_data, weights)+bias
        
    return out

def add(name, a, b):
    with tf.variable_scope(name):
        c = tf.add(a,b, name="1")

    return c

def model_pp(tensor_togther, name):
    with tf.variable_scope(name):
        x_2k_and_1 = []
        x_2k_and_1 = tf.split(tensor_togther, tensor_togther.get_shape()[3], 3)
        print('进入模型')
        CNN_name = 'CNN'
        K = len(x_2k_and_1)
        print('HHHHH', K)
        names = locals()
        for i in range(K):
            print('!!!!!!!!!', i)
            x_CNN = x_2k_and_1[i]
            print('输入到CNN中的张量是', x_CNN)
            names['CNN_out' + str(i) ] = double_CNNs(x_CNN, CNN_name)
            shape = names['CNN_out' + str(i) ].get_shape().as_list() 
            names['CNN_out' + str(i) ] = tf.reshape(names['CNN_out' + str(i) ], 
                                             [shape[0], shape[1], 1],name="reshaped")
            
        CNN_out = tf.concat([names['CNN_out' + str(0)], names['CNN_out' + str(1)]],2) 
        for i in range(2, K):
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>', i)
            CNN_out = tf.concat([CNN_out, names['CNN_out' + str(0)]], 2)
            print(CNN_out.get_shape)
        
        print('LSTM的输入是：', CNN_out)
        y_lstm1 = lstm_(CNN_out, name = 'lstm1')
        y_lstm1_drop = tf.nn.dropout(y_lstm1, keep_prob=0.5, name='y_lstm_drop1')
        print('第一个LSTM的输出是：', y_lstm1_drop)
        y_lstm2 = lstm_(y_lstm1_drop, 'lstm2')
        y_lstm2_drop = tf.nn.dropout(y_lstm2, keep_prob=0.5, name='y_lstm_drop2')
        print('第二个LSTM的输出是：', y_lstm2_drop)
        Y = tf.reshape(y_lstm2_drop, [y_lstm2_drop.get_shape()[0],-1])
        print(Y)
        y_ann = ann('ann', Y)
        print('BBBBBBBBBBBBBBB', y_ann)
        y = tf.nn.softmax(y_ann)
    return y
# =============================================================================
# 
# #CNN_name = 'XxXc' 
# x = tf.placeholder(tf.float32, [64,3000,1,1])
# #y = CNN(x, CNN_name, False)
# #print(y)
# 
# X_in = tf.placeholder(tf.float32, [64,3000,1, 5])
# Y = model_pp(X_in, 'mmm')
# print(Y)
#         
#         
# =============================================================================
        
        
        
        
        
        
        
        
        
        
        
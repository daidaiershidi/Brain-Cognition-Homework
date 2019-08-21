# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 19:45:23 2019

@author: 姜海洋
"""

# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

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
        
def lstm(name, input_data, seq_length, frame_size, flag=None):
    with tf.variable_scope(name):
        x = tf.reshape(input_data,shape=[-1,seq_length,frame_size], name="1")
        x = tf.transpose(x,[1,0,2], name="2")
        x = tf.reshape(x,shape=[-1,frame_size], name="3")
        x = tf.split(x,seq_length, name="4")
    
        lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size) # 正向RNN,输出神经元数量为128
     
        lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size) # 反向RNN,输出神经元数量为128
     
        output, fw_state, bw_state = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
        if flag == None:
            h = output[int(seq_length/2)]
        else:  
            weights=tf.get_variable("weights", [2*hidden_size,n_classes])#因为是双向，输出形状为（-1，2*hidden_num）
            bias=tf.get_variable("bias",[n_classes])
            h = tf.matmul(output[int(seq_length/2)],weights)+bias#output长度为sequence_length，我们取中间位置的输出，双向的结果都可以兼顾到
    
    return h

def ann(name, input_data):
    with tf.variable_scope(name):
        weights=tf.get_variable("weights", [1024,n_classes])
        bias=tf.get_variable("bias",[n_classes])  
        out = tf.matmul(input_data, weights)+bias
        
    return out

def add(name, a, b):
    with tf.variable_scope(name):
        c = tf.add(a,b, name="1")

    return c

def forward(x, two_ch, name, reuse):
    with tf.variable_scope(name, reuse=reuse):
        if two_ch:
            channel = 2
        else:
            channel = 1
        print('channel', channel)
        y1_conv1 = conv_1d('conv1_1', x, [Fs/2, 1, channel, 64], Fs/16, wd=1e-3)
        y1_pool1 = max_pool_1d('pool1_1', y1_conv1, 8, 8)
        y1_drop1 = tf.nn.dropout(y1_pool1, keep_prob=0.5, name='drop1_1')
        y1_conv2 = conv_1d('conv1_2', y1_drop1, [8,1,64,128], 1)
        y1_conv3 = conv_1d('conv1_3', y1_conv2, [8,1,128,128], 1)
        y1_conv4 = conv_1d('conv1_4', y1_conv3, [8,1,128,128], 1)  
        y1_ = add('add1',y1_conv2,y1_conv4)
        y1_pool2 = max_pool_1d('pool1_2', y1_, 4, 4)
    #    print(y1_conv1)
    #    print(y1_pool1)
    #    print(y1_drop1)
    #    print(y1_conv2)
    #    print(y1_conv3)
    #    print(y1_conv4)
    #    print(y1_pool2)
    #    print("###############################################3")
        y2_conv1 = conv_1d('conv2_1', x, [Fs*4, 1, channel, 64], Fs/2, wd=1e-3)
        y2_pool1 = max_pool_1d('pool2_1', y2_conv1, 4, 4)
        y2_drop1 = tf.nn.dropout(y2_pool1, keep_prob=0.5, name='drop2_1')
        y2_conv2 = conv_1d('conv2_2', y2_drop1, [8,1,64,128], 1)
        y2_conv3 = conv_1d('conv2_3', y2_conv2, [8,1,128,128], 1)
        y2_conv4 = conv_1d('conv2_4', y2_conv3, [8,1,128,128], 1) 
        y2_ = add('add2',y2_conv2,y2_conv4)
        y2_pool2 = max_pool_1d('pool2_2', y2_, 2, 2)
    #    print(y2_conv1)
    #    print(y2_pool1)
    #    print(y2_drop1)
    #    print(y2_conv2)
    #    print(y2_conv3)
    #    print(y2_conv4)
    #    print(y2_pool2)
    #    print("###############################################3")
        
        y_drop_ = tf.concat([y1_pool2,y2_pool2], 1)
        y_drop = tf.nn.dropout(y_drop_, keep_prob=0.5, name='drop')
        drop_shape = y_drop.get_shape().as_list() 
        nodes = drop_shape[1] * drop_shape[2] * drop_shape[3] 
        reshaped = tf.reshape(y_drop, [drop_shape[0], nodes],name="reshaped2fc") 
    #    print(y_drop)
        print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\', reshaped)
        
        y_fc = fc(name="y_fc", input_var=reshaped, n_hiddens=1024, bias=None, wd=0)
    #    print(y_fc)
        
        y_lstm1 = lstm('lstm1',reshaped,32,96)
        y_lstm1_drop = tf.nn.dropout(y_lstm1, keep_prob=0.5, name='y_lstm_drop1')
        y_lstm2 = lstm('lstm2',y_lstm1_drop,32,32)
        y_lstm2_drop = tf.nn.dropout(y_lstm2, keep_prob=0.5, name='y_lstm_drop2')
    #    print(y_lstm1_drop)
    #    print(y_lstm2_drop)
        
        y_add = add('add',y_lstm2_drop,y_fc)
    #    y_add = tf.concat([y_lstm1_drop, y_lstm1_drop], 1)
        y_add_drop = tf.nn.dropout(y_add, keep_prob=0.5, name='y_add_drop')  
    #    print(y_add_drop)
        y_ann = ann('ann', y_add_drop)
        #print('BBBBBBBBBBBBBBB', y_ann)
        y = tf.nn.softmax(y_ann)
        #print('AAAAAAAAAAAAAAAAAA', y)
    
    return y
# =============================================================================
# 
# x = tf.placeholder(tf.float32, [63,3000,1,1])
# channel = 1
# y = forward(x, channel, 'jkddd')
# print(y)
# 
# =============================================================================








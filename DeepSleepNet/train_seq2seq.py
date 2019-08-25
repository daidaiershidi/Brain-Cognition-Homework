# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 00:36:14 2019

@author: 姜海洋
"""

import tensorflow as tf
import os
import numpy as np
from get_data_sequence import two_ch_batch, one_ch_batch, get_doubel_file
from get_data_sequence import get_data, split_data_to_30s, K_EEG_together
from sklearn.model_selection import KFold
from model2 import model_pp



data_dir = './sleep-edf-database-expanded-1.0.0/sleep-cassette'
tf.reset_default_graph()

MODEL_SAVE_PATH = "./Models"
MODEL_NAME = 'model'

lr =  0.001
n_class = 5
batch_size = 64
input_dim = 30*100#30s*100
STEPS = 20
channel = 2
two_ch = False
Kfold = 3

def pre_data(data, k):
    
    X_train = []
    X_test = []
    
    data_len = len(data)
    kf = KFold(n_splits=k,shuffle=False)
    for train_index , test_index in kf.split(data):
        for i in train_index:
            X_train.append(data[i])
        for j in test_index:
            X_test.append(data[j])   
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    train_len = int(data_len*(k-1)/k)
    test_len = int(data_len/k)
#    print('$$$$$$$$$$$$$$$',X_train.shape, X_test.shape, train_len, test_len)
    return X_train, X_test, train_len, test_len
 
def backward():
    k = 2
    x = tf.placeholder(tf.float32, [batch_size,input_dim,1,2*k+1])
    y_ = tf.placeholder(tf.int32, [batch_size, n_class])
#    y = model.forward(x, two_ch, 'add', False)
    y = model_pp(x, 'model')
        #model.forward(x, two_ch, 'agdhd', False)
    global_step = tf.Variable(0, trainable=False) 
        
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))

    train_step = tf.train.AdamOptimizer(
        learning_rate=lr,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        name="Adam"
    ).minimize(loss)     

    correct_predition = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_predition,tf.float32)) 

    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        
        coord=tf.train.Coordinator()  
        threads= tf.train.start_queue_runners(coord=coord) 
        
        init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
        sess.run(init_op)
        
        #第一次训练模型注释掉以下两行
#        model_path=os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
#        tf.train.Saver().restore(sess,model_path)
        
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        
        
        epochs = 10
        
        for epoch in range(epochs):
            print('###########################################################################')
            doubel_file_name = get_doubel_file(data_dir)
#            print(doubel_file_name[0:3])
            X_train, X_test, train_len, test_len = pre_data(doubel_file_name, Kfold)
            acc1 = []
            loss1 = []
            for j in range(Kfold):            
                d = 0
#train
                train_loss = []
                for doubel_file in X_train[j*train_len:(j+1)*train_len]:
                    print('第%d轮第%dfold训练数据，第%d个人的数据：' % (epoch,j,d))
                    d = d+1
                    
                    file_1 = doubel_file[0]
                    file_2 = doubel_file[1]
                    EEG_1, EEG_2, onset, duration, stage_code = get_data(file_1, file_2, data_dir)
                    EEG1_sequence, EEG2_sequence, stage_sequence = split_data_to_30s(EEG_1
                                                        , EEG_2, onset, duration, stage_code)
                    if k!=0:
                        EEG1_sequence, EEG2_sequence, stage_sequence = K_EEG_together(EEG1_sequence,
                                                                EEG2_sequence, stage_sequence, k)
                
                    batch_time = len(EEG1_sequence)/batch_size
                    print('!!!!!!!训练中!!!!!!!!!!!!!!!!!!!!!!!!!', d-1, '\n')
                    
                    batch_data, batch_label = one_ch_batch(EEG1_sequence, 
                                                               stage_sequence, batch_size, k)
                    print('TTTTTTTTTT', batch_data.shape)
                    _, loss_value= sess.run([train_step, loss], 
                                                        feed_dict= {x:batch_data, y_:batch_label})
                    train_loss.append(loss_value)
                train_loss = sum(train_loss)/len(train_loss)
#test            
                test_acc = []
                for doubel_file in X_test[j*test_len:(j+1)*test_len]:
                    print('第%d轮第%dfold测试数据，第%d个人的数据：' % (epoch,j,d))
                    d = d+1
                    
                    file_1 = doubel_file[0]
                    file_2 = doubel_file[1]
                    EEG_1, EEG_2, onset, duration, stage_code = get_data(file_1, file_2, data_dir)
                    EEG1_sequence, EEG2_sequence, stage_sequence = split_data_to_30s(EEG_1
                                                        , EEG_2, onset, duration, stage_code)
                    if k!=0:
                        EEG1_sequence, EEG2_sequence, stage_sequence = K_EEG_together(EEG1_sequence,
                                                                EEG2_sequence, stage_sequence, k)
                
                    batch_time = len(EEG1_sequence)/batch_size
                    print('!!!!!!!测试中!!!!!!!!!!!!!!!!!!!!!!!!!', d-1, '\n')
                    for i in range(int(batch_time)):
                        if two_ch:
                            batch_data, batch_label = two_ch_batch(EEG1_sequence, 
                                                       EEG2_sequence, stage_sequence, batch_size)
                            #最后返回 batch_data, batch_label
                        else:
                            batch_data, batch_label = one_ch_batch(EEG1_sequence, 
                                                                   stage_sequence, batch_size)
                            
                        accuracy1 = sess.run([accuracy], 
                                             feed_dict= {x:batch_data, y_:batch_label})
                        test_acc.append(accuracy1[0])
                test_acc1 = sum(test_acc)/len(test_acc)
#
                loss1.append(train_loss)
                acc1.append(test_acc1)
            loss1 = sum(loss1)/len(loss1)
            acc1 = sum(acc1)/len(acc1)
            print('###########################################################################')
            print('########第%d轮kfold:loss is %g......acc is %f' % (epoch, loss1, acc1))
        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
def main():
    backward()
if __name__ =='__main__':
    main()

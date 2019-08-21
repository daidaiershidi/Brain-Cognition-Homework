# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 17:00:01 2019

@author: 姜海洋
"""

import tensorflow as tf
import os
import numpy as np
from get_data_sequence import two_ch_batch, one_ch_batch, get_doubel_file, before_batch
from get_data_sequence import get_data, split_data_to_30s, K_EEG_together, get_doubel_file_from_txt
from sklearn.model_selection import KFold
import model



data_dir = './sleep-edf-database-expanded-1.0.0/sleep-cassette'
tf.reset_default_graph()

MODEL_SAVE_PATH = "./Models"
MODEL_NAME = 'model'

lr =  0.00001
n_class = model.n_classes
batch_size = 64
input_dim = 30*100#30s*100
STEPS = 20
channel = 2
two_ch = False


def pre_data(data):
    
    data_train = data[33:]
    data_test = data[0:32]
#    print('$$$$$$$$$$$$$$$',X_train.shape, X_test.shape, train_len, test_len)
    return data_train, data_test
 
def backward():
    if two_ch:
        x = tf.placeholder(tf.float32, [batch_size,input_dim,1,2])
    else:
        x = tf.placeholder(tf.float32, [batch_size,input_dim,1,1])
    print('$$$$$$$$$$$$$$ x placeholder:', x)
    y_ = tf.placeholder(tf.int32, [batch_size, n_class])
#    y = model.forward(x, two_ch, 'add', False)
    y = model.forward(x, two_ch, 'adxd', False)
        #model.forward(x, two_ch, 'agdhd', False)
    global_step = tf.Variable(0, trainable=False) 
        
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
    
        tf.summary.scalar('loss', loss)

    learning_rate = tf.train.exponential_decay(
        lr,
        global_step,
        2048/64, 
        0.99,
        staircase=True)

    train_step = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        name="Adam"
    ).minimize(loss, global_step=global_step)     

    with tf.name_scope('accuracy'):
        correct_predition = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_predition,tf.float32)) 
        tf.summary.scalar('accuracy', accuracy)


    saver = tf.train.Saver()
    
    merged = tf.summary.merge_all()
    
    with tf.Session() as sess:
        
        coord=tf.train.Coordinator()  
        threads= tf.train.start_queue_runners(coord=coord) 
        
        init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
        sess.run(init_op)
        
        #第一次训练模型注释掉以下两行
        #model_path=os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
        #tf.train.Saver().restore(sess,model_path)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        k = 0
        m=0
        epoch = 0
        loss_epoch = []
        acc_test_epoch = []
        writer = tf.summary.FileWriter("./train", sess.graph)
        while 1:
            print('###########################################################################')
            #doubel_file_name = get_doubel_file(data_dir)
            doubel_file_name = get_doubel_file_from_txt('doubel_name_train.txt')
#            print(doubel_file_name[0:3])
            X_train, X_test = pre_data(doubel_file_name)
            acc1 = []
            loss1 = []
            epoch = epoch+1
            d = 0
#train
            train_loss = []
            for doubel_file in X_train:
                print('第%d轮训练数据，第%d个人的数据：' % (epoch,d))
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
                for i in range(int(batch_time)):
                    if two_ch:
                        batch_data, batch_label = two_ch_batch(EEG1_sequence, 
                                                   EEG2_sequence, stage_sequence, batch_size)
                        #最后返回 batch_data, batch_label
                    else:
                        batch_data, batch_label = one_ch_batch(EEG1_sequence, 
                                                               stage_sequence, batch_size)
                        
                    summary, _, loss_value= sess.run([merged, train_step, loss], 
                                                        feed_dict= {x:batch_data, y_:batch_label})
                    summary, _, loss_value,acc, y_pre = sess.run([merged, train_step, loss, accuracy, y], 
                                                    feed_dict= {x:batch_data, y_:batch_label})
                    writer.add_summary(summary, m)
                    writer.close()
                    m = m+1
                    train_loss.append(loss_value)
            train_loss = sum(train_loss)/len(train_loss)
            saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
#test       
            test_acc = []
            for doubel_file in X_test:
                print('第%d轮测试数据，第%d个人的数据：' % (epoch,d))
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
            loss_epoch.append(loss1)
            acc_test_epoch.append(acc1)
            print('###########################################################################')
            print('########第%d轮:loss is %g......acc is %f' % (epoch, loss1, acc1))
            if epoch>3:
#                if abs(loss_epoch[-1]-loss_epoch[-2])<=0.001 and abs(loss_epoch[-2]-loss_epoch[-3])<=0.001:
#                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
#                    break
#                if abs(acc_test_epoch[-3]-acc_test_epoch[-2])>=0.5 and abs(acc_test_epoch[-2]-acc_test_epoch[-1])>=0.5:
#                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
#                    print(max(acc-acc_test_epoch))
#                    break
                if acc_test_epoch[-4]>acc_test_epoch[-3] and acc_test_epoch[-3]>acc_test_epoch[-2] and acc_test_epoch[-2]>acc_test_epoch[-1]:
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
                    print(max(acc_test_epoch))
                    break
                
def main():
    backward()
if __name__ =='__main__':
    main()

    
    
    
    
    
    
    
    
    
    
    
    

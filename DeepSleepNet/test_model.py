# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 20:42:42 2019

@author: 姜海洋
"""
import tensorflow as tf
import os
import numpy as np
from get_data_sequence import two_ch_batch, one_ch_batch, get_doubel_file, before_batch
from get_data_sequence import get_data, split_data_to_30s, K_EEG_together, get_doubel_file_from_txt
from sklearn.model_selection import KFold
import model



data_dir = './sleep-edf-database-expanded-1.0.0/sleep-telemetry'
tf.reset_default_graph()

MODEL_SAVE_PATH = "./Models"
MODEL_NAME = 'model'

lr =  0.001
n_class = model.n_classes
batch_size = 64
input_dim = 30*100#30s*100
STEPS = 20
channel = 2
two_ch = False

def test_model():
    x = tf.placeholder(tf.float32, [batch_size,input_dim,1,1])
    y_ = tf.placeholder(tf.int32, [batch_size, n_class])
    y = model.forward(x, two_ch, 'adxd', False)
    global_step = tf.Variable(0, trainable=False) 
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
    
        tf.summary.scalar('loss', loss)
        
    correct_predition = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_predition,tf.float32)) 

    saver = tf.train.Saver()
    
    merged = tf.summary.merge_all()
    
    with tf.Session() as sess:
        
        coord=tf.train.Coordinator()  
        threads= tf.train.start_queue_runners(coord=coord) 
        
        init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
        sess.run(init_op)
        
        #第一次训练模型注释掉以下两行
        model_path=os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
        tf.train.Saver().restore(sess,model_path)
        
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        k = 0
        m = 0
        writer = tf.summary.FileWriter("./train", sess.graph)
        
        print('#######################################################')
        doubel_file_name = get_doubel_file_from_txt('doubel_name_test.txt')
        d=0
        for doubel_file in doubel_file_name:
            print('第%d个人的数据：' % d)
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
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', d)
            for i in range(int(batch_time)):
                if two_ch:
                    batch_data, batch_label = two_ch_batch(EEG1_sequence, 
                                               EEG2_sequence, stage_sequence, batch_size)
                    #最后返回 batch_data, batch_label
                else:
                    batch_data, batch_label = one_ch_batch(EEG1_sequence, 
                                                           stage_sequence, batch_size)
                    
                summary, loss_value,acc, y_pre = sess.run([merged, loss, accuracy, y], 
                                                    feed_dict= {x:batch_data, y_:batch_label})
                writer.add_summary(summary, m)
                writer.close()
                m = m+1
             
                print('loss is %g......acc is %f' % (loss_value,acc))
                
def main():
    test_model()
if __name__ =='__main__':
    main()
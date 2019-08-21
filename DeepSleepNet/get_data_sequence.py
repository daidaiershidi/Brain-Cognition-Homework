# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 09:55:55 2019

@author: 姜海洋
"""

import pandas as pd
from mne.io import RawArray, read_raw_edf
import mne
import os
import operator
import tensorflow as tf
import numpy as np
import random

#将数据集文件中的PSG文件和Hypnogram文件配对
def get_doubel_file(data_path):
    
    data_file_name = os.listdir(data_path)
    doubel_file = []
    for i in range(0, len(data_file_name)-1, 2):
        #doubel = 
        doubel_file.append([data_file_name[i], data_file_name[i+1]])
    #print(len(doubel_file))
    doubel_file = np.array(doubel_file)
    #print(type(doubel_file[5, 1]))
    
    return doubel_file


data_dir = './sleep-edf-database-expanded-1.0.0/sleep-cassette'

#def get_doubel_file_to_txt(data_path):
    
# =============================================================================
# data_file_name = os.listdir(data_dir)
# f1 = open('doubel_name.txt', 'a')
# for line in data_file_name:
#     f1.write(line)
#     f1.write("\n")
# f1.close()
# =============================================================================
    
def get_doubel_file_from_txt(txt_name):
    
    fi=open(txt_name,'r')
    txt=fi.readlines()
    data_file_name=[]
    for w in txt:
        w=w.replace('\n','')
        data_file_name.append(w)
    #print(len(data_file_name))
    doubel_file = []
    for i in range(0, len(data_file_name)-1, 2):
        #doubel = 
        doubel_file.append([data_file_name[i], data_file_name[i+1]])
    #print(len(doubel_file))
    doubel_file = np.array(doubel_file)
    #print(type(doubel_file[5, 1]))
    
    return doubel_file
    
#doubel_file = get_doubel_file_from_txt('doubel_name.txt')
#print(doubel_file)
    

#将一对PSG文件和Hypnogram文件中的数据提取两通道EEG、每段数据的起始时间、睡眠阶段编码
def get_data(file_1, file_2, data_path):
    
    EEG_path = data_path + '/' + file_1
    target_path = data_path + '/' + file_2
    #print(EEG_path)
    print('#########', EEG_path)
    print('#########', target_path)
    rawEEG = read_raw_edf(EEG_path)
    rawtarget = mne.read_annotations(target_path)
    data = rawEEG.to_data_frame().values.T   #所有的数据
    EEG_1 = data[0, :]
    EEG_2 = data[1, :]
    onset = rawtarget.onset   #每段数据的开始时间
    duration = rawtarget.duration   #每段数据持续的时间
    stage = rawtarget.description    #每段数据对应的睡眠阶段
    print(type(stage))
    print(stage[1])
    stage_code = []
    for i in stage:
        stage_code.append(stages[i])
    stage_code = np.array(stage_code)
    
    return EEG_1, EEG_2, onset, duration, stage_code

#把从一对文件中获得的两通道EEG、每段数据的起始时间、睡眠阶段编码做成每30秒一小段
#将结果保存在列表中
def split_data_to_30s(EEG_1, EEG_2, onset, duration, stage_code):
    
    EEG1_sequence = []
    EEG2_sequence = []
    stage_sequence = []
    i = 0
    for stage_i in stage_code:
        start = int(onset[i]*100)
        end = int(start + duration[i]*100)
        EEG1_stage = EEG_1[start:end]
        EEG2_stage = EEG_2[start:end]
        for j in range(int(EEG1_stage.shape[0]/3000)):
            EEG1_sequence.append(EEG1_stage[j:j+3000])
            EEG2_sequence.append(EEG2_stage[j:j+3000])
            stage_sequence.append(stage_i)
        i = i+1
        
    return EEG1_sequence, EEG2_sequence, stage_sequence
    
    
def K_EEG_together(EEG1_sequence, EEG2_sequence, EEG_stage, k):
    #将感兴趣的片段和它前后K个片段拼接在一起，当然，损失掉了数据序列中的前K个和后K个数据
    K_EEG1_sequence = []
    for i in range(k, len(EEG1_sequence)-k, 1):
        EEG1_insert = EEG1_sequence[i]
        for j in range(k):  #拼接前、后K个
            EEG1_insert = np.concatenate([EEG1_sequence[i-j-1], EEG1_insert, EEG1_sequence[i+j+1]])
        K_EEG1_sequence.append(EEG1_insert)
        K_stage_sequence = EEG_stage[k:-k]
        
    K_EEG2_sequence = []
    for i in range(k, len(EEG2_sequence)-k, 1):
        EEG2_insert = EEG2_sequence[i]
        for j in range(k):  #拼接前、后K个
            EEG2_insert = np.concatenate([EEG2_sequence[i-j-1], EEG2_insert, EEG2_sequence[i+j+1]])
        K_EEG2_sequence.append(EEG2_insert)
    
    return K_EEG1_sequence, K_EEG1_sequence, K_stage_sequence


def same_stage_together(EEG_sequence, EEG_stage, stages_dict):
    #由于数据序列中的各个睡眠阶段的数据数量不同，将相同睡眠阶段的 数据放在同一个列表中
    #以便获取小批量数据时，从包含各个睡眠阶段数据的列表中获取相同数量的数据，使训练数据均衡
    #EEG_sequence, EEG_stage分别是数据序列和标签序列，其中的数据与标签一一对应
    names = locals()
    stages_num = len(stages_dict)
    for i in range(stages_num):
        names['stages_' + str(i)] = []
        
    for index, EEG_data in enumerate(EEG_sequence):
        #print('index', index)
        #print('EEG_data', EEG_data)
        #从EEG数据序列中的第一个开始遍历，某元素的标签与字典中的哪个编码相同，就把该元素放入对应的列表中
        i = 0
        for value in stages_dict.values():
            if operator.eq(list(EEG_stage[index]),value):
                names['stages_' + str(i)].append(EEG_data)
            i = i+1
    j = 0
    #将每个stage放入其对应列表的最后一个
    for value in stages_dict.values():
        names['stages_' + str(j)].append(value)
        j = j+1
    
    #把几个stage的列表放入一个列表中
    out_stages = []
    for i in range(stages_num):
        out_stages.append(names['stages_' + str(i)])
    
    return out_stages
        
def get_balance_batch(out_stages, batch_size):
    #从代表不同睡眠阶段数据的列表中随机获取小批量数据 
    #在这个小批量数据中，各个睡眠阶段的数据个数是相同的，以达到训练数据均衡
    #batch_size_data是获取到的一个小批量数据
    #batch_size_label是小批量数据对应的标签
    stages_number = len(out_stages)  #判断每个阶段是否有数据，如果没有数据，删除这一阶段
    #print('每个睡眠阶段的数据数量为：', len(out_stages[0]), len(out_stages[1]),
          #len(out_stages[2]), len(out_stages[3]), len(out_stages[4]))
    for j in range(stages_number):
        if j==stages_number-1:
            break
        if len(out_stages[j])==1:
            #print('第%d个种类的数据被删除' % j)
            del out_stages[j]
            
    stages_number = len(out_stages)
    batch_size_data = []
    batch_size_label = []
    get_num = int(batch_size/stages_number)   #每一种stage数据需要取得数据的数量
    dis = batch_size - get_num*stages_number
    #dis_index = np.sort(np.random.randint(len(new_stages), size=dis))
    dis_index = np.sort(np.array(random.sample(range(0,stages_number),dis)))
    for i in range(stages_number):  #在每一种stage数据中随机取数据，有时需要某些stage多一个数据
        if not all(list(dis_index-i)):
            get_num_ = get_num+1
        else:
            get_num_ = get_num
        #随机获取某stage数据中取出数据的索引
        get_index = np.random.randint(len(out_stages[i][0:-1]),size=get_num_)
        m = 0
        for index in get_index:
            batch_size_data.append(out_stages[i][index])
            batch_size_label.append(out_stages[i][-1])
            m = m+1
            
    batch_size_data = np.array(batch_size_data)
    batch_size_label = np.array(batch_size_label)
    
    return batch_size_data, batch_size_label

def Two_EEG_balance_batch(out_stages, out_stages1, batch_size):
    #从代表不同睡眠阶段数据的列表中随机获取小批量数据 
    #在这个小批量数据中，各个睡眠阶段的数据个数是相同的，以达到训练数据均衡
    #batch_size_data是获取到的一个小批量数据
    #batch_size_label是小批量数据对应的标签
    stages_number = len(out_stages)  #判断每个阶段是否有数据，如果没有数据，删除这一阶段
    #print('每个睡眠阶段的数据数量为：', len(out_stages[0]), len(out_stages[1]),
          #len(out_stages[2]), len(out_stages[3]), len(out_stages[4]))
    for j in range(stages_number):
        if j==stages_number-1:
            break
        if len(out_stages[j])==1:
            #print('第%d个种类的数据被删除' % j)
            del out_stages[j]
            del out_stages1[j]
    
    stages_number = len(out_stages)
    batch_size_EEG1 = []
    batch_size_EEG2 = []
    batch_size_label = []
    get_num = int(batch_size/stages_number)   #每一种stage数据需要取得数据的数量
    dis = batch_size - get_num*stages_number
    #dis_index = np.sort(np.random.randint(len(new_stages), size=dis))
    dis_index = np.sort(np.array(random.sample(range(0,stages_number),dis)))
    for i in range(stages_number):  #在每一种stage数据中随机取数据，有时需要某些stage多一个数据
        if not all(list(dis_index-i)):
            get_num_ = get_num+1
        else:
            get_num_ = get_num
        #随机获取某stage数据中取出数据的索引
        get_index = np.random.randint(len(out_stages[i][0:-1]),size=get_num_)
        
        m = 0
        for index in get_index:
            batch_size_EEG1.append(out_stages[i][index])
            batch_size_EEG2.append(out_stages1[i][index])
            batch_size_label.append(out_stages[i][-1])
            m = m+1
            
    batch_size_EEG1 = np.array(batch_size_EEG1)
    batch_size_EEG2 = np.array(batch_size_EEG2)
    batch_size_EEG1 = batch_size_EEG1.reshape(batch_size_EEG1.shape[0],batch_size_EEG1.shape[1],1,1)
    batch_size_EEG2 = batch_size_EEG1.reshape(batch_size_EEG2.shape[0],batch_size_EEG2.shape[1],1,1)
    batch_size_label = np.array(batch_size_label)
    batch_size_data = np.concatenate([batch_size_EEG1, batch_size_EEG2], -1)
    
    return batch_size_data, batch_size_label
    

def two_ch_batch(EEG1_sequence, EEG2_sequence, stage_sequence, batch_size):     
    
    EEG1_same_stages = same_stage_together(EEG1_sequence, stage_sequence, new_stages)
    EEG2_same_stages = same_stage_together(EEG2_sequence, stage_sequence, new_stages)
    batch_size_data, batch_label = Two_EEG_balance_batch(EEG1_same_stages, EEG2_same_stages, batch_size) 
           
    return batch_size_data, batch_label
            
def one_ch_batch(EEG1_sequence, stage_sequence, batch_size):
    
    EEG1_same_stages = same_stage_together(EEG1_sequence, stage_sequence, new_stages)
    batch_size_EEG1_data, batch_EEG1_label = get_balance_batch(EEG1_same_stages, batch_size)
    batch_size_EEG1_data = batch_size_EEG1_data.reshape(batch_size_EEG1_data.shape[0], 
                                                                    batch_size_EEG1_data.shape[1], 1, 1)
    
    return batch_size_EEG1_data, batch_EEG1_label
    
def before_batch(data_dir, k):
    #doubel_file_name = get_doubel_file(data_dir)
    doubel_file_name = get_doubel_file_from_txt('doubel_name.txt')
    print(doubel_file_name)
    i=0
    for doubel_file in doubel_file_name:
        print('第%d个人的数据：' % i)
        i = i+1
        file_1 = doubel_file[0]
        file_2 = doubel_file[1]
        print('WWWWWWWWWWWWW', file_1, file_2)
        EEG_1, EEG_2, onset, duration, stage_code = get_data(file_1, file_2, data_dir)
        EEG1_sequence, EEG2_sequence, stage_sequence = split_data_to_30s(EEG_1
                                            , EEG_2, onset, duration, stage_code)
        if k!=0:
            EEG1_sequence, EEG2_sequence, stage_sequence = K_EEG_together(EEG1_sequence,
                                                    EEG2_sequence, stage_sequence, k)
        
        return EEG1_sequence, EEG2_sequence, stage_sequence





stages = {'Sleep stage W':[1, 0, 0, 0, 0], #Sleep stage W
          'Sleep stage 1':[0, 1, 0, 0, 0], #Sleep stage 1
          'Sleep stage 2':[0, 0, 1, 0, 0], #Sleep stage 2
          'Sleep stage 3':[0, 0, 0, 1, 0], #Sleep stage 3/4
          'Sleep stage 4':[0, 0, 0, 1, 0], #Sleep stage 3/4
          'Sleep stage R':[0, 0, 0, 0, 1], #Sleep stage R
          'Movement time':[1, 1, 0, 0, 0], #Movement time
          'Sleep stage ?':[1, 0, 1, 0, 0]  #Sleep stage ?
          }

new_stages = {'stage1':[1, 0, 0, 0, 0], #Sleep stage W
              'stage2':[0, 1, 0, 0, 0], #Sleep stage 1
              'stage3':[0, 0, 1, 0, 0], #Sleep stage 2
              'stage4':[0, 0, 0, 1, 0], #Sleep stage 3/4
              'stage5':[0, 0, 0, 0, 1], #Sleep stage R
              }
# =============================================================================
# 
# #从数据集文件夹中获取成对的数据文件，做成列表，列表中的每个元素是一对数据文件名
# doubel_file_name = get_doubel_file(data_dir)
# 
# #for i in doubel_file_name:
#     #print(i[0])
#     #print(i[1])
# i = doubel_file_name[0]
# file_1 = i[0]
# file_2 = i[1]
# #从一对数据文件中获取数据，为：两个EEG，每段数据的起始时间，长度，编码标签
# EEG_1, EEG_2, onset, duration, stage_code = get_data(file_1, file_2, data_dir)
# #将数据分成每30秒的小段，每个小段有两个EEG,一个编码标签
# EEG1_sequence, EEG2_sequence, stage_sequence = split_data_to_30s(EEG_1, EEG_2, onset, duration, stage_code)
#                                         
# #将感兴趣的片段和其前后各k个片段拼接在一起，当然，这样做损失掉了最前边、最后边的k个数据（此操作可以没有）
# k = 2
# #K_EEG1_sequence, K_EEG2_sequence, K_stage_sequence = K_EEG_together(EEG1_sequence, EEG2_sequence, stage_sequence, k)
# 
# #将数据序列中stage相同的数据放到一个列表中，列表的最后一个元素为该列表数据对应的标签
# #再将含有几个stage的列表放入一个列表中
# #new_stages是已经删除不需要数据后的类别
# EEG1_same_stages = same_stage_together(EEG1_sequence, stage_sequence, new_stages)
# EEG2_same_stages = same_stage_together(EEG2_sequence, stage_sequence, new_stages)
# print('EEEEEEEEEEEEEEEEEEEE', len(EEG1_same_stages))
# #获取种类均衡的小批量数据
# batch_size = 64
# batch_size_EEG1_data, batch_EEG1_label = get_balance_batch(EEG1_same_stages, batch_size)
# #batch_size_EEG2_data, batch_EEG2_label = get_balance_batch(EEG2_same_stages, batch_size)
# print('!!!!!!!!!!!!!!!!!')
# print('batch_size_label shape is \n', batch_EEG1_label)
# # =============================================================================
# # print(batch_size_EEG1_data.shape) 
# # #print(batch_size_EEG2_data.shape)     
# # batch_size_EEG1_data = batch_size_EEG1_data.reshape(batch_size_EEG1_data.shape[0], batch_size_EEG1_data.shape[1], 1, 1)
# # print('batch_size_EEG1_data shape after reshape', batch_size_EEG1_data.shape)
# # batch_size_EEG2_data = batch_size_EEG2_data.reshape(batch_size_EEG2_data.shape[0], batch_size_EEG2_data.shape[1], 1, 1)
# # print('batch_size_EEG1_data shape after reshape', batch_size_EEG2_data.shape)
# # 
# # 
# # batch_size_data, batch_label = Two_EEG_balance_batch(EEG1_same_stages, EEG1_same_stages, batch_size)
# # print('batch_size_data, batch_label', batch_size_data.shape, batch_label.shape)
# # 
# # 
# # 
# # =============================================================================
# =============================================================================








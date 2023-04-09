#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import pandas as pd
from random import shuffle
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy import sparse
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from collections import  defaultdict


# In[ ]:


import numpy as np
import math


# In[ ]:


from model import *


# In[ ]:


from utils import *


# In[ ]:


def getelements(s):
  list1 = []
  for i in range(len(s)):
    str1 = s[i]
    str2 = str1.replace("[","").replace("]","")
    str2 = str2.strip().split(",")
    element_list = []
    for j in range(len(str2)):
        element_list.append(int(str2[j]))
    list1.append(element_list)
  
  return list1
def compute_corr(prob_seq, next_seq, corr_dic):
    corr= np.zeros((prob_seq.shape[0],prob_seq.shape[1], prob_seq.shape[1]))
    for i in range(0,prob_seq.shape[0]):
        for  j in range(0,next_seq.shape[1] ):
            for k in range(j+1):
                corr[i][j][k]=corr_dic[next_seq[i][j]][prob_seq[i][k]]
    return corr


# In[ ]:


def prepare_batches_test(train_data, batch_size, randomize=True):
    """Prepare batches grouping padded sequences.

    Arguments:
        data (list of lists of torch Tensor): output by get_data
        batch_size (int): number of sequences per batch

    Output:
        batches (list of lists of torch Tensor)
    """
    # if randomize:
    #     shuffle(train_data)
    batches = []
    train_y, train_problem, timestamps, train_real_len = train_data["y"], train_data["problem"], train_data["timestamp"], train_data["real_len"]
    train_y = np.array(getelements(train_y))
    train_problem = np.array(getelements(train_problem))
    timestamps = np.array(getelements(timestamps))
    item_ids = [torch.LongTensor(i) for i in train_problem]
    timestamp = [torch.LongTensor(timestamp) for timestamp in timestamps]
    labels = [torch.LongTensor(i) for i in train_y]
    item_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), i))[:-1] for i in item_ids]
    # skill_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), s))[:-1] for s in skill_ids]
    label_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), l))[:-1] for l in labels]
    data = list(zip(item_inputs, label_inputs, item_ids, timestamp, labels))
    
    #划分相应的batch

    for k in range(0, len(data), batch_size):
        batch = data[k:k + batch_size]
        seq_lists = list(zip(*batch))

        inputs_and_ids = [pad_sequence(seqs, batch_first=True, padding_value=0)
                          for seqs in seq_lists[:-1]]
        labels = pad_sequence(seq_lists[-1], batch_first=True, padding_value=-1)  # Pad labels with -1
        batches.append([*inputs_and_ids, labels])

    return batches


# In[ ]:


def prepare_batches(train_data, batch_size, randomize=True):
    """Prepare batches grouping padded sequences.

    Arguments:
        data (list of lists of torch Tensor): output by get_data
        batch_size (int): number of sequences per batch

    Output:
        batches (list of lists of torch Tensor)
    """
    # if randomize:
    #     shuffle(train_data)
    batches = []
    train_y, train_problem, timestamps, train_real_len = train_data[0], train_data[1], train_data[2], train_data[3]

    item_ids = [torch.LongTensor(i) for i in train_problem]
    timestamp = [torch.LongTensor(timestamp) for timestamp in timestamps]
    labels = [torch.LongTensor(i) for i in train_y]
    item_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), i))[:-1] for i in item_ids]
    # skill_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), s))[:-1] for s in skill_ids]
    label_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), l))[:-1] for l in labels]
    data = list(zip(item_inputs, label_inputs, item_ids, timestamp, labels))
    
    #划分相应的batch

    for k in range(0, len(data), batch_size):
        batch = data[k:k + batch_size]
        seq_lists = list(zip(*batch))

        inputs_and_ids = [pad_sequence(seqs, batch_first=True, padding_value=0)
                          for seqs in seq_lists[:-1]]
        labels = pad_sequence(seq_lists[-1], batch_first=True, padding_value=-1)  # Pad labels with -1
        batches.append([*inputs_and_ids, labels])

    return batches

def train_test_split(data, split=0.8):
    n_samples = data[0].shape[0]
    split_point = int(n_samples*split)
    train_data, test_data = [], []
    for d in data:
        train_data.append(d[:split_point])
        test_data.append(d[split_point:])
    return train_data, test_data
def compute_auc(preds, labels):
    preds = preds[labels >= 0].flatten()
    labels = labels[labels >= 0].float()
    if len(torch.unique(labels)) == 1:  # Only one class
        auc = accuracy_score(labels, preds.round())
        acc = auc
    else:
        auc = roc_auc_score(labels, preds)
        acc = accuracy_score(labels, preds.round())
    return auc, acc

def compute_loss(preds, labels, criterion,partial_loss):
    preds = preds[labels >= 0].flatten()
    labels = labels[labels >= 0].float()
    return criterion(preds, labels) + partial_loss
def computeRePos(time_seq, time_span):
    batch_size = time_seq.shape[0]
    size = time_seq.shape[1]

    time_matrix= (torch.abs(torch.unsqueeze(time_seq, axis=1).repeat(1,size,1).reshape((batch_size, size*size,1)) - 
                            torch.unsqueeze(time_seq,axis=-1).repeat(1, 1, size,).reshape((batch_size, size*size,1))))

    # time_matrix[time_matrix>time_span] = time_span
    time_matrix = time_matrix.reshape((batch_size,size,size))
    return (time_matrix)
def get_corr_data(pro_num):
    pro_pro_dense = np.zeros((pro_num, pro_num))
    list1 = []
    tmp = []
    pro_pro_ = open('./datasets/exercise_matrix.txt',"r")
    for i in pro_pro_:
        j = i.strip().split(",")
        tmp = [float(str(k)) for k in j]
        list1.append(tmp)
    pro_pro_dense = np.array(list1)
    return pro_pro_dense


# In[ ]:


def train_loss(preds,labels):
    preds = preds[labels >= 0].flatten()
    labels = labels[labels >= 0].float()
    criterion = nn.BCEWithLogitsLoss()
    total_loss =  criterion(preds,labels) 
    return total_loss
    


# In[ ]:


def get_q_matrix(embed_size):
    with open("./datasets/q_matrix.txt","r",encoding='utf-8') as f:
        q_matrix = []
        for line in f.readlines():
            token = line.strip().split(" ")
            list1 = [int(float(i)) for i in token]
            tmp_len = len(list1)
            while tmp_len < embed_size:
                list1.append(0)
                tmp_len= tmp_len + 1
            q_matrix.append(list1)
    return q_matrix


# In[ ]:


def train2(train_data, val_data, pro_num, timestamp, timespan,  model, optimizer, logger, saver,q_matrix, lamda, beta, num_epochs, batch_size, grad_clip):
    
    step = 0
    metrics = Metrics()
    corr_data = get_corr_data(pro_num)
    result_train = []
  
    for epoch in range(num_epochs):
        train_batches = prepare_batches(train_data, batch_size)
        val_batches = prepare_batches(val_data, batch_size)
        tmp_train = []
        for item_inputs, label_inputs, item_ids, timestamp, labels in train_batches: 
            rel = corr_data[(item_ids-1).unsqueeze(1).repeat(1,item_ids.shape[-1],1),(item_inputs-1).unsqueeze(-1).repeat(1,1,item_inputs.shape[-1])]
            item_inputs = item_inputs.cuda()
            time = computeRePos(timestamp, timespan)
            label_inputs = label_inputs.cuda()
            item_ids = item_ids.cuda()
            preds, weights = model(q_matrix, item_inputs, label_inputs, item_ids, torch.Tensor(rel), time)
            loss = train_loss(preds,labels.cuda())
            preds = torch.sigmoid(preds).detach().cpu()
            train_auc, train_acc = compute_auc(preds, labels)
            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            step += 1
            metrics.store({'loss/train': loss.item()})
            metrics.store({'auc/train': train_auc})
            if step == len(train_batches)-1:
                torch.save(weights, 'weight_tensor_rel')
            if step % 1000 == 0:
                logger.log_scalars(metrics.average(), step)
                
            model.eval()
            
        for item_inputs, label_inputs, item_ids, timestamp, labels in val_batches:
            rel = corr_data[
                (item_ids - 1).unsqueeze(1).repeat(1, item_ids.shape[-1], 1), (item_inputs - 1).unsqueeze(-1).repeat(1,
                                                                      1,
                                                                      item_inputs.shape[-1])]
            item_inputs = item_inputs.cuda()
            time = computeRePos(timestamp, timespan)
            label_inputs = label_inputs.cuda()
            item_ids = item_ids.cuda()
            with torch.no_grad():
                preds,weights = model(q_matrix,item_inputs, label_inputs, item_ids, torch.Tensor(rel).cuda(), time.cuda())
                preds = torch.sigmoid(preds).cpu()
                val_auc, val_acc = compute_auc(preds, labels)
                metrics.store({'auc/val': val_auc, 'acc/val': val_acc})

        model.train()
        average_metrics = metrics.average()
        print(average_metrics)
        stop = saver.save(average_metrics['auc/val'], model)
        if stop:  
            break


# In[ ]:


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train NGFKT.')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--logdir', type=str, default='runs/ngfkt')
    parser.add_argument('--savedir', type=str, default='save/ngfkt')
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--embed_size', type=int, default=200)
    parser.add_argument('--num_attn_layers', type=int, default=1) 
    parser.add_argument('--num_heads', type=int, default=5)
    parser.add_argument('--encode_pos')  
    parser.add_argument('--max_pos', type=int, default=10)   
    parser.add_argument('--drop_prob', type=float, default=0.2)  
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--test_size', type=int, default=10) 
    parser.add_argument('--lr', type=float, default=1e-2)  
    parser.add_argument('--grad_clip', type=float, default=10)  
    parser.add_argument('--num_epochs', type=int, default=300) 
    parser.add_argument('--timespan', default=100000, type=int)
    parser.add_argument('--lamda', default=0.1, type=float)
    parser.add_argument('--beta', default=0.1, type=float)
    args = parser.parse_args(args=[])
    print("Start to load data",flush = True)
    data = pd.read_csv("./datasets/train_Eedi.csv")
    y, problem, timestamp, real_len = data['y'], data['problem'], data['timestamp'] , data['real_len']
    y = np.array(getelements(y))
    problem = np.array(getelements(problem))
    timestamp = np.array(getelements(timestamp))
    real_len = np.array(real_len)
    skill_num = 86
    pro_num = 948
    q_matrix = get_q_matrix(args.embed_size)
    print('problem number %d, skill number %d' % (pro_num, skill_num),flush = True)
    print("divide train test set",flush = True)
    train_data, test_data1 = train_test_split([y, problem, timestamp, real_len])
    num_items = pro_num
    print("Get NGFKT",flush = True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NGFKT(num_items, args.embed_size, args.num_attn_layers, args.num_heads,
                  args.encode_pos, args.max_pos, args.drop_prob).cuda()
    optimizer = Adam(model.parameters(), lr=args.lr)
    while True:
        param_str = (f'{args.dataset},'
                     f'batch_size={args.batch_size},'
                     f'max_length={args.max_length},'
                     f'encode_pos={args.encode_pos},'
                     f'max_pos={args.max_pos}')
        logger = Logger(os.path.join(args.logdir, param_str))
        saver = Saver(args.savedir, param_str)
        train2(train_data, test_data1, pro_num, timestamp, args.timespan, model, optimizer, logger, saver, q_matrix, args.lamda,args.beta, args.num_epochs,
              args.batch_size, args.grad_clip)
        break
    logger.close()
    param_str = (f'{args.dataset},'
                  f'batch_size={args.batch_size},'
                  f'max_length={args.max_length},'
                  f'encode_pos={args.encode_pos},'
                  f'max_pos={args.max_pos}')
    saver = Saver(args.savedir, param_str)
    
    test_set  = pd.read_csv("./datasets//test_Eedi.csv")
    y, problem, timestamp, real_len = test_set['y'], test_set['problem'], test_set['timestamp'] , test_set['real_len']
    test_batches = prepare_batches_test(test_set, args.test_size, randomize=True)
    
    corr_data = get_corr_data(num_items)
    test_preds = np.empty(0)
    model = saver.load()
    # Predict on test set
    model.eval()
    correct = np.empty(0)
    acc = []
    auc = []
    print("Testing")
    
    count = 0
    for item_inputs, label_inputs, item_ids, timestamp, labels in test_batches:
        rel = corr_data[
                (item_ids - 1).unsqueeze(1).repeat(1, item_ids.shape[-1], 1), (item_inputs - 1).unsqueeze(-1).repeat(1,
                                                                                                                     1,
                                                                                                            item_inputs.shape[
                                                                                                                         -1])]
        item_inputs = item_inputs.cuda()
            # skill_inputs = skill_inputs.cuda()
        time = computeRePos(timestamp, args.timespan)
        label_inputs = label_inputs.cuda()
        item_ids = item_ids.cuda()
            # skill_ids = skill_ids.cuda()
        with torch.no_grad():
                preds,weights = model(q_matrix,item_inputs, label_inputs, item_ids, torch.Tensor(rel).cuda(), time.cuda())
                preds = torch.sigmoid(preds).cpu()
        test_auc, test_acc = compute_auc(preds, labels)
        auc.append(test_auc)
        acc.append(test_acc)
    print("The auc on test is {}. The acc on test is {}".format(np.mean(auc),np.mean(acc)))   
    


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# In[2]:


def future_mask(seq_length):
    #向上第一个对角线保留数据
    future_mask = np.triu(np.ones((1, seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)


def clone(module, num):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])


# In[3]:


# rel 应该是relation的缩写
def attention(query, key, value, rel, l1, l2,timestamp, clip_factor, guessing_factor, mask=None, dropout=None):
    """Compute scaled dot product attention.
    """
    #rel = rel * mask.to(torch.float) # future masking of correlation matrix.
    rel = torch.Tensor(rel).to(torch.float)
    rel_attn = rel.masked_fill(rel == 0, -10000)
    rel_attn = nn.Softmax(dim=-1)(rel_attn)
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / math.sqrt(query.size(-1))
    clip_factor = clip_factor(key)
    guessing_factor = guessing_factor(key)
    R1_factor = torch.sigmoid(guessing_factor-clip_factor) + guessing_factor
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)

        time_stamp= torch.exp(-torch.abs(timestamp.float()))
        #np.inf 表示没有确切的数，但是该数的类型为浮点型
        time_stamp=time_stamp.masked_fill(mask,-np.inf)
        
        Rs_factor = time_stamp+ R1_factor

 
    prob_attn = F.softmax(scores, dim=-1)
    time_attn = F.softmax(Rs_factor,dim=-1)
    prob_attn = (1-l2)*prob_attn+l2*time_attn
    # prob_attn = F.softmax(prob_attn + rel_attn, dim=-1)

    prob_attn = (1-l1)*prob_attn + (l1)*rel_attn
    
    
    if dropout is not None:
        prob_attn = dropout(prob_attn)
    return torch.matmul(prob_attn, value), prob_attn


# In[4]:


# the relationsip based attention model
def relative_attention(query, key, value, rel, l1, l2, timestamp,
                        pos_key_embeds, pos_value_embeds, mask=None, dropout=None):
    
    assert pos_key_embeds.num_embeddings == pos_value_embeds.num_embeddings
    
    scores = torch.matmul(query, key.transpose(-2, -1))
    #利用arange重新生成一个数组
    idxs = torch.arange(scores.size(-1))
    if query.is_cuda:
        idxs = idxs.cuda()
    idxs = idxs.view(-1, 1) - idxs.view(1, -1)
    idxs = torch.clamp(idxs, 0, pos_key_embeds.num_embeddings - 1)
    # aij_k
    pos_key = pos_key_embeds(idxs).transpose(-2, -1)
    pos_scores = torch.matmul(query.unsqueeze(-2), pos_key)
    #在倒数第二行加一个维度
    scores = scores.unsqueeze(-2) + pos_scores
    #算出了eij
    scores = scores / math.sqrt(query.size(-1))
    
    #aij_v
    pos_value = pos_value_embeds(idxs)
    value = value.unsqueeze(-3) + pos_value

    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(-2), -1e9)
    #compute the a_ij
    prob_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        prob_attn = dropout(prob_attn)
        
    #output Zij

    output = torch.matmul(prob_attn, value).unsqueeze(-2)
    prob_attn = prob_attn.unsqueeze(-2)
    return output, prob_attn
    


# In[7]:


class MultiHeadedAttention(nn.Module):
    def __init__(self, total_size, num_heads, drop_prob,clip_factor, guessing_factor):
        super(MultiHeadedAttention, self).__init__()
        assert total_size % num_heads == 0
        self.total_size = total_size
        # the number of heads
        self.head_size = total_size // num_heads
        self.num_heads = num_heads
        # three layers
        self.linear_layers = clone(nn.Linear(total_size, total_size), 3)
        self.dropout = nn.Dropout(p=drop_prob)
        self.clip_factor = clip_factor
        self.guessing_factor = guessing_factor
        
    
    def forward(self, query, key, value, rel, l1, l2,timestamp, encode_pos, 
                pos_key_embeds, pos_value_embeds, mask=None):
        batch_size, seq_length = query.shape[:2]

        # Apply mask to all heads
        if mask is not None:
            mask = mask.unsqueeze(1)

        # Project inputs
        '''
        前面提到multihead attention需要4个linear layer,而下面这个代码用到了其中的三个
        最后的一个linear 是self.linear[-1],
        下面这个代码等效于
        query, key, value = [l(x) for l, x in zip(self.linears, (query, key, value))]
        query,  key, value = [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                     for x in (query, key, value)]
        第一行先把QKV进行了linear变换，然后第二行将QKV维向量分解为h*d_k
        
        '''
        rel = rel.unsqueeze(1).repeat(1,self.num_heads,1,1)
        timestamp = timestamp.unsqueeze(1).repeat(1,self.num_heads,1,1)
        query, key, value = [l(x).view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        '''
        # Apply attention
        if encode_pos:
            out, self.prob_attn = relative_attention(
                query, key, value, rel, l1, l2, timestamp, pos_key_embeds, pos_value_embeds,  mask, self.dropout)
        else:
            out, self.prob_attn = attention(query, key, value, rel, l1, l2, timestamp, mask, self.dropout)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.total_size)
        return out, self.prob_attn
        '''
        if encode_pos :
            out, self.prob_attn = relative_attention(
                query, key, value, rel, l1, l2, timestamp, pos_key_embeds, pos_value_embeds,  mask, self.dropout)
        else:
            out, self.prob_attn = attention(query, key, value, rel, l1, l2,  
                                                  timestamp, self.clip_factor, self.guessing_factor, mask=None, dropout=None)
        
        
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.total_size)
        
        return out, self.prob_attn
    
    
   
       
    
  


# In[ ]:





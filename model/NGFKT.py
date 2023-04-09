#!/usr/bin/env python
# coding: utf-8

# In[4]:



from .attention import *
# In[7]:


import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# In[8]:
def future_mask(seq_length):
    #向上第一个对角线保留数据
    future_mask = np.triu(np.ones((1, seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)


def clone(module, num):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])


class NGFKT(nn.Module):
    def __init__(self, num_items,  embed_size, num_attn_layers, num_heads,
                 encode_pos, max_pos, drop_prob):
        """Self-attentive knowledge tracing.
        Arguments:
            num_items (int): number of items
            num_skills (int): number of skills
            embed_size (int): input embedding and attention dot-product dimension
            num_attn_layers (int): number of attention layers
            num_heads (int): number of parallel attention heads
            encode_pos (bool): if True, use relative position embeddings
            max_pos (int): number of position embeddings to use
            drop_prob (float): dropout probability
        """
        super(NGFKT, self).__init__()
        self.embed_size = embed_size
        self.encode_pos = encode_pos

        self.item_embeds = nn.Embedding(num_items + 1, embed_size , padding_idx=0)
        # self.skill_embeds = nn.Embedding(num_skills + 1, embed_size // 2, padding_idx=0)

        self.pos_key_embeds = nn.Embedding(max_pos, embed_size // num_heads)
        self.pos_value_embeds = nn.Embedding(max_pos, embed_size // num_heads)
        clip_factor = nn.Embedding(embed_size, embed_size)
        guessing_factor = nn.Embedding(embed_size, embed_size)
        self.clip_factor = clip_factor
        self.guessing_factor = guessing_factor
        self.lin_in = nn.Linear(2*embed_size, embed_size)
        self.attn_layers = clone(MultiHeadedAttention(embed_size, num_heads, drop_prob, clip_factor,guessing_factor), num_attn_layers)
        self.dropout = nn.Dropout(p=drop_prob)
        self.lin_out = nn.Linear(embed_size, 1)
        self.l1 = nn.Parameter(torch.rand(1))
        self.l2 = nn.Parameter(torch.rand(1))
    
    def get_knowledge_embedding(self, skill_nums, item_inputs,embed_size,q_matrix):
        
        if skill_nums >1000:
            return self.item_embeds(item_inputs)
        
        knowledge_embed = torch.zeros((len(item_inputs),len(item_inputs[0]),embed_size))
        for i in range(len(item_inputs)):
            for j in range(len(item_inputs[i])):
                if j==0:
                    a = torch.zeros((1,embed_size))
                    knowledge_embed[i][j] = a
                else:
                    index = item_inputs[i][j].int()
                    embed = torch.tensor(q_matrix[index])
                    knowledge_embed[i][j] = embed
            
        return knowledge_embed
    
    def get_inputs(self, skill_nums, q_matrix,item_inputs, label_inputs):
        item_inputs = self.get_knowledge_embedding(skill_nums, item_inputs,self.embed_size,q_matrix)
        item_inputs = item_inputs.cuda()
        
        # skill_inputs = self.skill_embeds(skill_inputs)
        label_inputs = label_inputs.unsqueeze(-1).float()
        

        inputs = torch.cat([item_inputs, item_inputs], dim=-1)
        inputs[..., :self.embed_size] *= label_inputs
        inputs[..., self.embed_size:] *= 1 - label_inputs
        return inputs

    def get_query(self, item_ids):
        item_ids = self.item_embeds(item_ids)
        # skill_ids = self.skill_embeds(skill_ids)
        query = torch.cat([item_ids], dim=-1)
        return query

    def forward(self, skill_nums, q_matrix, item_inputs, label_inputs, item_ids, rel, timestamp):
       
        inputs = self.get_inputs(skill_nums, q_matrix,item_inputs, label_inputs)

        inputs = F.relu(self.lin_in(inputs))

        query = self.get_query(item_ids)

        mask = future_mask(inputs.size(-2))
        if inputs.is_cuda:
            mask = mask.cuda()
        '''
        self, query, key, value, rel, l1, l2, timestamp, encode_pos, pos_key_embeds, pos_value_embeds, mask=None
        
        '''
        outputs, attn  = self.attn_layers[0](query, inputs, inputs, rel, self.l1, self.l2, timestamp, True,
                                                   self.pos_key_embeds, self.pos_value_embeds, mask)
        outputs = self.dropout(outputs)
        
        for l in self.attn_layers[1:]:
            residual, attn = l(query, outputs, outputs, rel, self.l1, self.l2, False, timestamp,self.pos_key_embeds,
                         self.pos_value_embeds, mask)
            outputs = self.dropout(outputs + F.relu(residual))

        return self.lin_out(outputs), attn


# In[ ]:





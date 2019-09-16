# -*- coding: utf-8 -*-
# @Author: micheala
# @Created: 12/09/19 
# @Contact: michealabaho265@gmail.com
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size, K_size, Q_size, method=None):
        super(Attention, self).__init__()

        K_size = 2*hidden_size if K_size is None else K_size
        Q_size = hidden_size if Q_size is None else Q_size
        self.method = method
        #given a bidirectional encoder
        self.key = nn.Linear(K_size, hidden_size)
        self.query = nn.Linear(Q_size, hidden_size)
        self.V = nn.Parameter(torch.FloatTensor(hidden_size))

    def foward(self, query, input, encoder_outputs):
        query_in = self.query(query)

        if self.method == 'dot':
            scores = torch.sum(query * encoder_outputs, dim=2)
        elif self.method == 'general':
            energy = self.query(encoder_outputs)
            scores = torch.sum(query * energy, dim=2)
        elif self.method == 'concat_score':
            energy = torch.tanh(torch.cat((query, encoder_outputs), 1))
            energy = self.key(energy)
            scores = torch.sum(self.V * energy, dim=2)
        elif self.method == 'bahdanau':
            proj_key = self.key(encoder_outputs)
            scores = self.energy(torch.tanh(query_in + proj_key))

        attn_weights = F.softmax(scores, dim=1)

        return  attn_weights
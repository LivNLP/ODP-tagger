# -*- coding: utf-8 -*-
# @Author: micheala
# @Created: 16/09/19 
# @Contact: michealabaho265@gmail.com

import torch
import torch.nn as nn
import torch.nn.functional as F
import data_prep as dp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiLstm_model(nn.Module):
    def __init__(self, embedding_dim, hidden_size, vocab_size, output_size, dropout_rate, weights_tensor=None, use_pretrained_embeddings=False):
        super(BiLstm_model, self).__init__()

        self.hidden_size = hidden_size
        if use_pretrained_embeddings:
            weights = [list(i.numpy())[:embedding_dim] for i in weights_tensor]
            weights = torch.FloatTensor(weights)
            self.embedding = nn.Embedding.from_pretrained(weights)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.maphiddenToOutput = nn.Linear(2*hidden_size, output_size)
        self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, bidirectional=True)

    def init_hidden(self):
        #cater for the bidirection in the LSTM
        return (torch.zeros(2, 1, self.hidden_size, device=device),
                torch.zeros(2, 1, self.hidden_size, device=device))

    def forward(self, input_seq):
        embedding = self.dropout(self.embedding(input_seq))
        lstm_out, self.hidden = self.lstm(embedding.view(len(input_seq), 1, -1), self.hidden)
        lstm_dout = self.dropout(lstm_out)
        dense_layer = self.maphiddenToOutput(lstm_dout.view(len(input_seq), -1))
        class_scores = F.log_softmax(dense_layer, dim=1)
        return class_scores

if __name__=="__main__":
    file_path = 'ebm-data/train_ebm.bmes'
    word_map, word_count, index2word = dp.create_vocabularly(file_path)
    line_pairs, outputs = dp.readwordTag(file_path)
    encoded_outputs = dp.encod_outputs(outputs)
    x = BiLstm_model(50, 64, len(word_map), len(encoded_outputs))
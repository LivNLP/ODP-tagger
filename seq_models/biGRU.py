# -*- coding: utf-8 -*-
# @Author: micheala
# @Created: 16/09/19
# @Contact: michealabaho265@gmail.com

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import data_prep as dp
import numpy as np

class BiGRU_model(nn.Module):
    def __init__(self, embedding_dim, hidden_size, vocab_size, pos_size, tag_map, dropout_rate, pos=False, weights_tensor=None, use_pretrained_embeddings=False, crf=False):
        super(BiGRU_model, self).__init__()
        self.outputsize = len(tag_map)
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.pos = pos
        self.use_pretrained_embeddings = use_pretrained_embeddings
        if self.use_pretrained_embeddings:
            # weights_tensor = weights_tensor.cpu() if weights_tensor.type().__contains__('cuda') else weights_tensor
            weights = [list(i.numpy())[:embedding_dim] for i in weights_tensor]
            self.weights = torch.FloatTensor(weights).to(dp.device)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.pos_embedding = nn.Embedding(pos_size, embedding_dim)
        self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(dropout_rate)
        self.pos_gru = nn.GRU(int(embedding_dim), hidden_size, bidirectional=True)
        self.word_gru = nn.GRU(int(embedding_dim), hidden_size, bidirectional=True)
        self.gru= nn.GRU(input_size=hidden_size*2 if self.pos else embedding_dim, hidden_size=hidden_size, bidirectional=True)
        if crf:
            self.maphiddenToOutput = nn.Linear(2*hidden_size, self.outputsize+2)
        else:
            self.maphiddenToOutput = nn.Linear(2*hidden_size, self.outputsize)

    def init_hidden(self):
        #cater for the bidirection in the LSTM
        return (torch.zeros(2, 1, self.hidden_size, device=dp.device))

    def forward(self, input_seq, pos_seq, hidden):
        if self.use_pretrained_embeddings:
            word_embedding = self.obtain_seq_embedding(input_seq, self.weights)
        else:
            word_embedding = self.dropout(self.embedding(input_seq))
        wgru_out, _ = self.word_gru(word_embedding.view(len(input_seq), 1, -1), self.hidden)

        pos_embedding = self.dropout(self.pos_embedding(pos_seq))
        pgru_out, _ = self.pos_gru(pos_embedding.view(len(pos_seq), 1, -1), self.hidden)

        word_pos = torch.add(wgru_out, pgru_out)
        word_pos = self.dropout(word_pos)
        if self.pos:
            out_put, hidden = self.gru(word_pos.view(len(input_seq), 1, -1), hidden)
        else:
            out_put, hidden = self.gru(word_embedding.view(len(input_seq), 1, -1), hidden)
        f_out_put = self.dropout(out_put)
        dense_layer = self.maphiddenToOutput(f_out_put.view(len(input_seq), -1))
        return dense_layer

    def obtain_seq_embedding(self, seq, pretrained_emb):
        seq_len = len(seq)
        embs = torch.zeros((seq_len, 1, self.embedding_dim), device=dp.device)
        for i,j in enumerate(seq):
            embs[i] = pretrained_emb[j]
        return embs

    def forward_batch(self, input_seq, pos_seq, hidden):
        #padding
        sent_lens = sorted([len(i) for i in input_seq], reverse=True)
        sentences = sorted(input_seq, key=lambda x:len(x), reverse=True)
        print(sentences[:2])
        pad_sentences = pad_sequence(sentences, batccath_first=True)
        print(pad_sentences[:2])
        print(sent_lens)
        packed_words = pack_padded_sequence(pad_sentences, sent_lens, batch_first=True)
        packed_words = self.dropout(packed_words)
        output, hidden = self.lstm(packed_words, hidden)
        f_output = self.dropout(output)
        dense_layer = self.maphiddenToOutput(f_output.view(len(input_seq), -1))
        return dense_layer




if __name__=="__main__":
    file_path = 'ebm-data/train_ebm.bmes'
    word_map, word_count, index2word = dp.create_vocabularly(file_path)
    line_pairs, outputs = dp.readwordTag(file_path)
    encoded_outputs = dp.encod_outputs(outputs)
    x = BiLstm_model(50, 64, len(word_map), len(encoded_outputs))
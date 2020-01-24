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

class BiLstm_model(nn.Module):
    def __init__(self, embedding_dim, hidden_size, vocab_size, pos_size, tag_map, word_map, dropout_rate, pos=False, weights_tensor=None, use_pretrained_embeddings=False,
                 use_bert_features=False, crf=False):
        super(BiLstm_model, self).__init__()
        self.outputsize = len(tag_map)
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.pos = pos
        self.word_map = word_map
        self.use_pretrained_embeddings = use_pretrained_embeddings
        self.use_bert_features = use_bert_features
        if self.use_pretrained_embeddings and not self.use_bert_features:
            # weights_tensor = weights_tensor.cpu() if weights_tensor.type().__contains__('cuda') else weights_tensor
            weights = [list(i.numpy())[:embedding_dim] for i in weights_tensor]
            self.weights = torch.FloatTensor(weights).to(dp.device)
        elif self.use_pretrained_embeddings and self.use_bert_features:
            self.weights = weights_tensor
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.pos_embedding = nn.Embedding(pos_size, embedding_dim)
        self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(dropout_rate)
        self.pos_lstm = nn.LSTM(int(embedding_dim), hidden_size, bidirectional=True)
        self.word_lstm = nn.LSTM(int(embedding_dim), hidden_size, bidirectional=True)
        self.lstm = nn.LSTM(input_size=hidden_size*2 if self.pos else embedding_dim, hidden_size=hidden_size, bidirectional=True)
        if crf:
            self.maphiddenToOutput = nn.Linear(2*hidden_size, self.outputsize+2)
        else:
            self.maphiddenToOutput = nn.Linear(2*hidden_size, self.outputsize)

    def init_hidden(self):
        #cater for the bidirection in the LSTM
        return (torch.zeros(2, 1, self.hidden_size, device=dp.device),
                torch.zeros(2, 1, self.hidden_size, device=dp.device))

    def forward(self, input_seq, pos_seq, hidden):
        if self.use_pretrained_embeddings:
            word_embedding = self.dropout(self.obtain_seq_embedding(input_seq, self.weights))
        else:
            word_embedding = self.dropout(self.embedding(input_seq))
        wlstm_out, _ = self.word_lstm(word_embedding.view(len(input_seq), 1, -1), self.hidden)
        #print('wlstm_out', wlstm_out.size())
        pos_embedding = self.dropout(self.pos_embedding(pos_seq))
        plstm_out, _ = self.pos_lstm(pos_embedding.view(len(pos_seq), 1, -1), self.hidden)

        word_pos = torch.add(wlstm_out, plstm_out)
        word_pos = self.dropout(word_pos)
        if self.pos:
            out_put, hidden = self.lstm(word_pos.view(len(input_seq), 1, -1), hidden)
        else:
            out_put, hidden = self.lstm(word_embedding.view(len(input_seq), 1, -1), hidden)
        f_out_put = self.dropout(out_put)
        dense_layer = self.maphiddenToOutput(f_out_put.view(len(input_seq), -1))
        return dense_layer

    def obtain_seq_embedding(self, input_seq, features):
        seq_len = len(input_seq)
        embs = torch.zeros((seq_len, 1, self.embedding_dim), device=dp.device)
        if self.use_bert_features:
            input_seq = [i.item() for i in input_seq]
            for i in features:
                words = [w[0] for w in features[i]]
                instance = [self.word_map[i] for i in words]
                if input_seq == instance:
                    for k, v in enumerate(features[i]):
                        if type(v[1]) != torch.Tensor:
                            embs[k] = torch.tensor(v[1][:self.embedding_dim])
                        else:
                            embs[k] = v[1][:self.embedding_dim]

        else:
            for i,j in enumerate(input_seq):
                embs[i] = features[j]

        return embs

    def forward_batch(self, input_seq, pos_seq, hidden):
        #padding
        sent_lens = sorted([len(i) for i in input_seq], reverse=True)
        sentences = sorted(input_seq, key=lambda x:len(x), reverse=True)
        pad_sentences = pad_sequence(sentences, batccath_first=True)
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
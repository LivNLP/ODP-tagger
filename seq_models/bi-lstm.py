import torch
import torch.nn as nn
import torch.nn.functional as F
import data_prep as dp


class BiLstm_model(nn.Module):
    def __init__(self, embedding_dim, hidden_size, vocab_size, output_size, dropout_rate):
        super(BiLstm_model, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.maphiddenToOutput = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, bidirectional=True, num_layers=1)

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_size, device=dp.device))

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
    x = Lstm_model(50, 64, len(word_map), len(encoded_outputs))
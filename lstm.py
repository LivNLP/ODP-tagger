import torch
import torch.nn as nn
import torch.nn.functional as F
import data_prep as dp

class Lstm_model(nn.Module):
    def __init__(self, embedding_dim, hidden_size, vocab_size, output_size):
        super(Lstm_model, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size)
        self.maphiddenToOutput = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq):
        embedding = self.embedding(input_seq)
        lstm_out, (lstm_hiddenstate, lstm_cellstate) = self.lstm(embedding.view(len(input_seq), 1, -1))
        dense_layer = self.maphiddenToOutput(lstm_out.view(len(input_seq), -1))
        class_scores = F.log_softmax(dense_layer, dim=1)
        return class_scores

if __name__=="__main__":
    file_path = 'ebm-data/train_ebm.bmes'
    word_map, word_count, index2word = dp.create_vocabularly(file_path)
    line_pairs, outputs = dp.readwordTag(file_path)
    encoded_outputs = dp.encod_outputs(outputs)
    x = Lstm_model(50, 64, len(word_map), len(encoded_outputs))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class outcomeEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_size, vocab_size, n_layers=1):
        super(outcomeEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size)

        #self.map2output = nn.Linear(hidden_size, num_classes)

    def forward(self, input, hidden):
        c_hidden = hidden
        embedding = self.embeddings(input).view(1, 1, -1)
        output, enc_hidden = self.lstm(embedding, (hidden, c_hidden))
        #tag_space = self.map2output(lstm_out.view(len(input), -1))
        #tag_scores = F.log_softmax(tag_space, dim=1)
        return output, enc_hidden[0]

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class outcomeDecoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(outcomeDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        c_hidden = hidden
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, dec_hidden = self.lstm(output, (hidden, c_hidden))
        output = self.softmax(self.out(output[0]))
        return output, dec_hidden[0]

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
        
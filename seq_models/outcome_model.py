import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class outcomeEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_size, vocab_size, bidirectional=False):
        super(outcomeEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        if bidirectional:
            self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, bidirectional=True)
        else:
            self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size)

    def forward(self, input, hidden):
        c_hidden = hidden
        embedding = self.embeddings(input).view(1, 1, -1)
        output, enc_hidden = self.lstm(embedding, (hidden, c_hidden))
        return output, enc_hidden[0]

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class outcomeDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout, n_layers, bidirectional=False):
        super(outcomeDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(output_size, hidden_size)
        if bidirectional:
            self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=n_layers, bidirectional=True)
        else:
            self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=n_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        d_hidden = hidden
        input_embedding = self.embedding(input).view(1, 1, -1)
        input_embedding = self.dropout(input_embedding)
        input_embedding = F.relu(input_embedding)
        output, dec_hidden = self.lstm(input_embedding, (hidden, d_hidden))
        output = self.softmax(self.out(output[0]))
        return output, dec_hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class outcomeAttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout, n_layers, max_length):
        super(outcomeAttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = nn.Dropout(dropout)
        self.max_length = max_length

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attn = nn.Linear(hidden_size*2, max_length)
        self.attn_combine = nn.Linear(hidden_size*2, hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=n_layers)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        ad_hidden = hidden
        embedded = self.embedding(input).view(1, 1, -1)
        input_embedding = self.dropout(embedded)

        input_transformed = self.attn(torch.cat((embedded[0], hidden[0]), 1))
        attn_weights = F.softmax(input_transformed, dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, dec_hidden = self.lstm(output, (hidden, ad_hidden))

        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, dec_hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


        
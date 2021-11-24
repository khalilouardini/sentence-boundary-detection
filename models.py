import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_size, dropout_rate):
        """
        :param: dims: sequence input dimensions
        :param: lstm_dim: LSTM £hidden unit dimension
        :param: dropout_rate: dropout layer probability
        :return: None
        """
        super(LSTM, self).__init__()

        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, lstm_size, batch_first=True)
        self.fc = nn.Linear(lstm_size, 1)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        """
        :param: x: sequence input 
        :return: forward pass output
        """
        x = self.emb(x)
        #x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, (h, c) = self.lstm(x)
        x = self.fc(h)
        x = self.dropout(x)
        return torch.sigmoid(x)


class Bi_LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_size, dropout_rate):
        """
        :param: dims: sequence input dimensions
        :param: lstm_dim: LSTM £hidden unit dimension
        :param: dropout_rate: dropout layer probability
        :return: None
        """
        super(Bi_LSTM, self).__init__()

        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, lstm_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(lstm_size*2, 1)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        """
        :param: x: sequence input 
        :return: forward pass output
        """
        x = self.emb(x)
        #x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, (h, c) = self.lstm(x)
        hidden = torch.cat((h[-2,:,:], h[-1,:,:]), dim = 1)
        x = self.fc(hidden)
        x = self.dropout(x)
        return torch.sigmoid(x)

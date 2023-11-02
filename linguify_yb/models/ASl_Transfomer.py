"""doc
"""
import torch.nn as nn
from torch import optim
import torch


class Token(nn.Module):
    def __init__(self, number_vocab=1000, max_len=100, number_hidden=64):
        super().__init__()
        self.postional_embedding_layers = nn.Embedding(number_vocab, number_hidden)
        self.embedding_layers = nn.Embedding(max_len, number_hidden)

    def forward(self, input_X):
        max_len = torch.tensor.size(input_X)[-1]
        input_X = self.embedding_layers(input_X)
        # Generate positions using torch.arange
        positions = torch.arange(0, max_len)
        positions = self.postional_embedding_layers(postions)
        return input_X + postions


class Landmark_Em(nn.Module):
    def __init__(self, number_hidden=64, max_len=100):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=number_hidden, out_channels=11, kernel_size= 2, padding="same", stride=2
        )
        self.conv2 = nn.Conv1d(
            in_channels=number_hidden, out_channels=11, padding="same", stride=2
        )
        self.conv3 = nn.Conv1d(
            in_channels=number_hidden, out_channels=11, padding="same", stride=2
        )
        self.postions_embedding_layers = nn.Embedding(maxlen, hidden_hidden)

    def forward(self, input_X):
        outputs = nn.Sequential(
            [self.conv1(), nn.ReLU(), self.conv2(), nn.ReLU(), self.conv3(), nn.ReLU()]
        )
        return outputs

class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, number_heads, feed_forward_dim, dropout_rate=0.1):
        super().__init__()
        self.multi_head_att = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads= number_heads)
        self.layer_norm_1 = nn.LayerNorm()
        self.layer_norm_2 = nn.LayerNorm()
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.fcn = nn.Sequential([nn.Linear(feed_forwrd_dim)],
                                            nn.ReLU(),
                                           nn.Linear(embedding_dim))
    def forward(self, input_X):
        multi_att_outputs = self.multi_head_att(input_X, input_X)
        multi_att_outputs = self.dropout_1(multi_att_outputs)
        outputs = self.layer_norm_1(input_X + multi_att_outputs)
        fcn_outputs = self.Sequential(outputs)
        fcn_outputs = self.dropout_2(fcn_ouputs)
        outputs = self.layer_norm_1(outputs + fcn_outputs)
        return outputs

class DecoderTransformer(nn.Module):
    def __int__():
        super().__int__()
        self.multi_head_att = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads= number_heads)
        self.layer_norm_1 = nn.LayerNorm()
        self.layer_norm_2 = nn.LayerNorm()
        self.layer_norm_3 = nn.LayerNorm()
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.fcn = nn.Sequential([nn.Linear(feed_forwrd_dim)],
                                            nn.ReLU(),
                                           nn.Linear(embedding_dim))
    
    def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
        pass
    def forward(self, input_x):
        pass

"""doc
"""

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(self, number_vocab=60 , max_len=100, embedding_dim=64):
        super().__init__()
        self.postional_embedding_layers = nn.Embedding(number_vocab, embedding_dim)
        self.embedding_layers = nn.Embedding(max_len, embedding_dim)

    def forward(self, input_x):
        max_len = input_x.size()[-1]
        input_x = self.embedding_layers(input_x)
        # Generate positions using torch.arange
        positions = torch.arange(0, max_len)
        positions = self.postional_embedding_layers(positions)
        return input_x + positions


class LandmarkEmbedding(nn.Module):
    def __init__(self, input_dim = None, number_hidden=64, max_len=100):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=number_hidden,
            kernel_size=11,
            padding="same",
            stride=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=number_hidden,
            out_channels=number_hidden,
            kernel_size=11,
            padding="same",
            stride=1,
        )
        self.conv3 = nn.Conv1d(
            in_channels=number_hidden,
            out_channels=number_hidden,
            kernel_size=11,
            padding="same",
            stride=1,
        )
        self.postions_embedding_layers = nn.Embedding(max_len, number_hidden)
        self.seq_nn = nn.Sequential(
            self.conv1, nn.ReLU(), self.conv2, nn.ReLU(), self.conv3, nn.ReLU()
        )

    def forward(self, input_x):
        outputs = self.seq_nn(input_x)
        return outputs


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        source_maxlen=100,
        target_maxlen=100,
        no_multi_heads=6,
    ):
        super().__init__()
        num_encoder_layers = num_decoder_layers = 6
        encoder_forward_dim = 100
        # Define encoder and decoder layers
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=no_multi_heads,
            dim_feedforward=encoder_forward_dim,
            activation="relu",
        )

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_dim,
            nhead=no_multi_heads,
            dim_feedforward=output_dim,
            activation="relu",
        )

        # Define encoder and decoder
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_encoder_layers
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layer, num_layers=num_decoder_layers
        )

        # Input and output linear layers
        self.input_linear = LandmarkEmbedding(input_dim=input_dim,max_len=source_maxlen)
        self.target_linear = TokenEmbedding(max_len=target_maxlen)
        self.num_classes = 60
        self.output_linear = nn.Linear(output_dim, self.num_classes)

    def forward(self, input_x, input_y):
        # Apply EMbedding
        input_x = self.input_linear(input_x)

        # Transformer encoding
        memory = self.transformer_encoder(input_x)

        # Apply linear layer to the target
        input_y = self.target_linear(input_y)

        # Transformer decoding
        output = self.transformer_decoder(input_y, memory)

        # Apply linear layer to the output
        output = self.output_linear(output)

        return output

    # TODO code generate for inference
    def generate(
        self,
    ):
        pass

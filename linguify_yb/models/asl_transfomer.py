"""doc
"""

import torch
from torch import nn


class Token(nn.Module):
    def __init__(self, number_vocab=1000, max_len=100, number_hidden=64):
        super().__init__()
        self.postional_embedding_layers = nn.Embedding(number_vocab, number_hidden)
        self.embedding_layers = nn.Embedding(max_len, number_hidden)

    def forward(self, input_x):
        max_len = torch.tensor.size(input_x)[-1]
        input_x = self.embedding_layers(input_x)
        # Generate positions using torch.arange
        positions = torch.arange(0, max_len)
        positions = self.postional_embedding_layers(positions)
        return input_x + positions


class Landmark_Em(nn.Module):
    def __init__(self, number_hidden=64, max_len=100):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=number_hidden,
            out_channels=11,
            kernel_size=2,
            padding="same",
            stride=2,
        )
        self.conv2 = nn.Conv1d(
            in_channels=number_hidden,
            out_channels=11,
            kernel_size=2,
            padding="same",
            stride=2,
        )
        self.conv3 = nn.Conv1d(
            in_channels=number_hidden,
            out_channels=11,
            kernel_size=2,
            padding="same",
            stride=2,
        )
        self.postions_embedding_layers = nn.Embedding(max_len, number_hidden)
        self.seq_nn = nn.Sequential(
            [self.conv1(), nn.ReLU(), self.conv2(), nn.ReLU(), self.conv3(), nn.ReLU()]
        )

    def forward(self, input_x):
        outputs = self.seq_nn(input_x)
        return outputs


class Transformer(nn.Module):
    def __init__(
        self, input_dim, output_dim, nhead, num_encoder_layers, num_decoder_layers
    ):
        super(Transformer, self).__init__()

        # Define encoder and decoder layers
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=nhead)

        # Define encoder and decoder
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_encoder_layers
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layer, num_layers=num_decoder_layers
        )

        # Input and output linear layers
        self.input_linear = nn.Linear(input_dim, input_dim)
        self.output_linear = nn.Linear(input_dim, output_dim)

    def forward(self, src, tgt):
        # Apply linear layer to the input
        src = self.input_linear(src)

        # Transformer encoding
        memory = self.transformer_encoder(src)

        # Apply linear layer to the target
        tgt = self.input_linear(tgt)

        # Transformer decoding
        output = self.transformer_decoder(tgt, memory)

        # Apply linear layer to the output
        output = self.output_linear(output)

        return output


# Example usage:
def test_func(Transformer_):
    input_dim = 512  # Adjust based on your input dimension
    output_dim = 256  # Adjust based on your output dimension
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6

    # Instantiate the model
    model = Transformer_(
        input_dim, output_dim, nhead, num_encoder_layers, num_decoder_layers
    )

    # Create dummy input
    src = torch.randn((10, 32, input_dim))  # (sequence_length, batch_size, input_dim)
    tgt = torch.randn((20, 32, input_dim))  # (sequence_length, batch_size, input_dim)

    # Forward pass
    output = model(src, tgt)

    # Print the output shape
    print("Output shape:", output.shape)


test_func(Transformer)

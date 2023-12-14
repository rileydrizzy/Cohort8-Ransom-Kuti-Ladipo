"""
Baseline Transformer Module

This module contains the implementation of a Transformer model for sign language tasks.

Classes:
- TokenEmbedding: Create embedding for the target seqeunce
- LandmarkEmbedding: Create embedding for the source(frames)seqeunce
- Encoder: Implements the transformer encoder stack.
- Decoder: Implements the transformer decoder stack.
- Transformer: The main transformer model class with methods for training and inference.

Methods:
- Transformer.generate: Perform inference on a new sequence
"""
import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """Embed the tokens with position encoding"""

    def __init__(self, num_vocab, maxlen, embedding_dim):
        """
        Parameters
        ----------
        num_vocab : int
            number of character vocabulary
        maxlen : int
            maximum length of sequence
        embedding_dim : int
            embedding output dimension
        """
        super().__init__()
        self.token_embed_layer = nn.Embedding(num_vocab, embedding_dim)
        self.position_embed_layer = nn.Embedding(maxlen, embedding_dim)

    def forward(self, x):
        """
        Parameters
        ----------
        x : tensors
            input tensor with shape (batch_size, sequence_length)

        Returns
        -------
        tensors
            embedded tensor with shape (batch_size, sequence_length, embedding_dim)
        """
        batch_size, maxlen = x.size()

        # Token embedding
        x = self.token_embed_layer(x)

        # Positional encoding
        positions = torch.arange(0, maxlen).to(x.device)
        positions = (
            self.position_embed_layer(positions).unsqueeze(0).expand(batch_size, -1, -1)
        )

        return x + positions


class LandmarkEmbedding(nn.Module):
    """_summary_"""

    def __init__(self, embedding_dim):
        super().__init__()
        # Calculate the padding for "same" padding
        padding = (11 - 1) // 2

        # Define three 1D convolutional layers with ReLU activation and stride 2
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=64, kernel_size=11, stride=2, padding=padding
        )
        self.conv2 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=11, stride=2, padding=padding
        )
        self.conv3 = nn.Conv1d(
            in_channels=128, out_channels=256, kernel_size=11, stride=2, padding=padding
        )

        # Output embedding layer
        self.embedding_layer = nn.Linear(256, embedding_dim)

    def forward(self, x):
        # Input x should have shape (batch_size, input_size, input_dim)
        x = x.unsqueeze(1)  # Add a channel dimension for 1D convolution

        # Apply convolutional layers with ReLU activation and stride 2
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # Global average pooling to reduce spatial dimensions
        x = torch.mean(x, dim=2)

        # Apply the linear embedding layer
        x = self.embedding_layer(x)

        return x


class TransformerEncoder(nn.Module):
    """Transformer Encoder Module"""

    def __init__(
        self,
        embedding_dim,
        num_heads,
        feed_forward_dim,
        dropout_rate=0.1,
    ):
        """Initialize the Transformer Encoder

        Parameters
        ----------
        embedding_dim : int
            Dimension of input embeddings
        num_heads : int
            Number of attention heads in the multi-head attention layer
        feed_forward_dim : int
            Dimension of the feed-forward layer
        dropout_rate : float, optional
            Dropout rate, by default 0.1
        """
        super().__init__()
        self.multi_attention = nn.MultiheadAttention(embedding_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, embedding_dim),
        )

        self.layernorm1 = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs_x):
        # Multi-head attention
        multi_attention_out, _ = self.multi_attention(inputs_x, inputs_x, inputs_x)
        multi_attention_out = self.dropout1(multi_attention_out)

        # Residual connection and layer normalization
        out1 = self.layernorm1(inputs_x + multi_attention_out)

        # Feed-forward layer
        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out)

        # Residual connection and layer normalization
        x = self.layernorm2(out1 + ffn_out)

        return x


class TransformerDecoder(nn.Module):
    """Transformer Decoder Module"""

    def __init__(self, embedding_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        """Initialize the Transformer Decoder

        Parameters
        ----------
        embedding_dim : int
            Dimension of input embeddings
        num_heads : int
            Number of attention heads in the multi-head attention layer
        feed_forward_dim : int
            Dimension of the feed-forward layer
        dropout_rate : float, optional
            Dropout rate, by default 0.1
        """
        super().__init__()
        self.num_heads_ = num_heads
        self.layernorm1 = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.decoder_multi_attention = nn.MultiheadAttention(
            embedding_dim, num_heads, batch_first=True
        )
        self.encoder_multi_attention = nn.MultiheadAttention(
            embedding_dim, num_heads, batch_first=True
        )
        self.decoder_dropout = nn.Dropout(0.5)
        self.encoder_dropout = nn.Dropout(dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, embedding_dim),
        )

    def _causal_attention_mask(self, sequence_length, batch_size, device=None):
        mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1).to(
            device
        )
        mask = mask.unsqueeze(0).expand(
            batch_size * self.num_heads_, sequence_length, sequence_length
        )
        return mask

    def forward(self, encoder_out, src_target):
        input_shape = src_target.size()
        batch_size, seq_len, _ = input_shape
        x_device = src_target.device

        # Mask
        causal_mask = self._causal_attention_mask(
            sequence_length=seq_len, batch_size=batch_size, device=x_device
        )

        target_att, _ = self.decoder_multi_attention(
            src_target, src_target, src_target, attn_mask=causal_mask
        )
        target_norm_out = self.layernorm1(src_target + self.decoder_dropout(target_att))
        encoder_out, _ = self.encoder_multi_attention(
            target_norm_out, encoder_out, encoder_out
        )
        enc_out_norm = self.layernorm2(encoder_out + self.encoder_dropout(encoder_out))

        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out))

        return ffn_out_norm


class ASLTransformer(nn.Module):
    def __init__(
        self,
        num_hidden_dim=64,
        multi_num_head=8,
        num_feed_forward=128,
        target_maxlen=64,
        num_layers_enc=4,
        num_layers_dec=4,
    ):
        """_summary_

        Parameters
        ----------
        num_hidden_dim : int, optional
            _description_, by default 64
        multi_num_head : int, optional
            _description_, by default 8
        num_feed_forward : int, optional
            _description_, by default 128
        target_maxlen : int, optional
            _description_, by default 64
        num_layers_enc : int, optional
            _description_, by default 4
        num_layers_dec : int, optional
            _description_, by default 4
        """
        super().__init__()
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.target_maxlen = target_maxlen
        self.num_classes = 62

        self.encoder_input = LandmarkEmbedding(embedding_dim=num_hidden_dim)
        self.decoder_input = TokenEmbedding(
            num_vocab=self.num_classes,
            embedding_dim=num_hidden_dim,
            maxlen=target_maxlen,
        )

        self.encoder = nn.Sequential(
            self.encoder_input,
            *[
                TransformerEncoder(
                    embedding_dim=num_hidden_dim,
                    num_heads=multi_num_head,
                    feed_forward_dim=num_feed_forward,
                )
                for _ in range(num_layers_enc)
            ],
        )

        for i in range(num_layers_dec):
            self.add_module(
                f"decoder_layer_{i}",
                TransformerDecoder(
                    embedding_dim=num_hidden_dim,
                    num_heads=multi_num_head,
                    feed_forward_dim=num_feed_forward,
                ),
            )

        self.classifier = nn.Linear(
            in_features=num_hidden_dim, out_features=self.num_classes
        )

    def forward(self, source, target):
        encoder_out = self.encoder(source)
        transformer_output = self._decoder_run(encoder_out, target)
        return self.classifier(transformer_output)

    def _decoder_run(self, enc_out, target):
        decoder_out = self.decoder_input(target)
        for i in range(self.num_layers_dec):
            decoder_out = getattr(self, f"decoder_layer_{i}")(enc_out, decoder_out)
        return decoder_out

    def generate(self, source, target_start_token_idx=60):
        """Performs inference over one batch of inputs using greedy decoding

        Parameters
        ----------
        source : _type_
            _description_
        target_start_token_idx : int
            _description_

        Returns
        -------
        _type_
            _description_
        """
        encoder_out = self.encoder(source)
        decoder_input = (
            torch.ones((1), dtype=torch.long).to(source.device) * target_start_token_idx
        )
        dec_logits = []
        for _ in range(self.target_maxlen - 1):
            decoder_out = self._decoder_run(encoder_out, decoder_input)
            logits = self.classifier(decoder_out)

            logits = torch.argmax(logits, dim=-1, keepdim=True)
            last_logit = logits[-1]
            dec_logits.append(last_logit)
            decoder_input = torch.cat([decoder_input, last_logit], dim=-1)
        return decoder_input

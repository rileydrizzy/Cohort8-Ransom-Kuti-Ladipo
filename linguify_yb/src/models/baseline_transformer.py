"""doc
"""

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(self, num_vocab=1000, maxlen=100, num_hid=200):
        super().__init__()
        self.emb = nn.Embedding(num_vocab, num_hid)
        self.pos_emb = nn.Embedding(maxlen, num_hid)

    def forward(self, x):
        maxlen = x.size(-1)
        x = self.emb(x)
        positions = torch.arange(0, maxlen).to(x.device)
        positions = self.pos_emb(positions)
        return x + positions


class LandmarkEmbedding(nn.Module):
    def __init__(self, num_hid=64, maxlen=100):
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
        self.embedding_layer = nn.Linear(256, num_hid)

    def forward(self, x):
        # Input x should have shape (batch_size, input_size)
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
    def __init__(
        self,
        embed_dim,
        num_heads,
        feed_forward_dim,
        rate=0.1,
    ):
        super().__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, embed_dim),
        )

        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, inputs):
        attn_out, _ = self.att(inputs, inputs, inputs)
        attn_out = self.dropout1(attn_out)
        out1 = self.layernorm1(inputs + attn_out)

        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out)
        x = self.layernorm2(out1 + ffn_out)
        print(f"endocder{x.shape}")
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.self_att = nn.MultiheadAttention(embed_dim, num_heads)
        self.enc_att = nn.MultiheadAttention(embed_dim, num_heads)
        self.self_dropout = nn.Dropout(0.5)
        self.enc_dropout = nn.Dropout(0.1)
        self.ffn_dropout = nn.Dropout(0.1)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, embed_dim),
        )

    def causal_attention_mask(
        self, sequence_length, batch_size=1, num_heads=4, device="cpu"
    ):
        mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1).to(
            device
        )
        mask = mask.unsqueeze(0).expand(
            batch_size * num_heads, sequence_length, sequence_length
        )
        return mask

    def forward(
        self,
        enc_out,
        src_target_,
    ):
        input_shape = src_target_.size()
        batch_size = 1  # input_shape[0]
        seq_len = input_shape[0]
        mask = self.causal_attention_mask(seq_len, batch_size=batch_size)
        target_att, _ = self.self_att(
            src_target_, src_target_, src_target_, attn_mask=mask
        )
        target_norm = self.layernorm1(src_target_ + self.self_dropout(target_att))

        enc_out, _ = self.enc_att(target_norm, enc_out, enc_out)
        enc_out_norm = self.layernorm2(enc_out + self.enc_dropout(enc_out))

        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out))
        print(f"decoder - {ffn_out_norm.shape}")
        return ffn_out_norm


class NTransformer(nn.Module):
    def __init__(
        self,
        num_hid=64,
        num_head=8,
        num_feed_forward=128,
        source_maxlen=100,
        target_maxlen=100,
        num_layers_enc=4,
        num_layers_dec=4,
    ):
        super().__init__()
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.target_maxlen = target_maxlen
        self.num_classes = 62

        self.enc_input = LandmarkEmbedding(num_hid=num_hid, maxlen=source_maxlen)
        self.dec_input = TokenEmbedding(
            num_vocab=self.num_classes, maxlen=target_maxlen
        )

        self.encoder = nn.Sequential(
            self.enc_input,
            *[
                TransformerEncoder(num_hid, num_head, num_feed_forward)
                for _ in range(num_layers_enc)
            ],
        )

        for i in range(num_layers_dec):
            self.add_module(
                f"dec_layer_{i}",
                TransformerDecoder(num_hid, num_head, num_feed_forward),
            )

        self.classifier = nn.Linear(num_hid, self.num_classes)

    def forward(self, source, target):
        x = self.encoder(source)
        y = self.decoder_run(x, target)
        print(y.shape)
        return self.classifier(y)

    def decoder_run(self, enc_out, target):
        print(f"before emb {target.shape}")
        y = self.dec_input(target)
        print(f"after emb {y.shape}")

        for i in range(self.num_layers_dec):
            y = getattr(self, f"dec_layer_{i}")(enc_out, y)
        return y

    def generate(self, source, target_start_token_idx=60):
        """Performs inference over one batch of inputs using greedy decoding

        Parameters
        ----------
        source : _type_
            _description_
        target_start_token_idx : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        bs = source.size(0)
        enc = self.encoder(source)
        dec_input = (
            torch.ones((1), dtype=torch.long).to(source.device) * target_start_token_idx
        )
        dec_logits = []
        for i in range(self.target_maxlen - 1):
            dec_out = self.decoder_run(enc, dec_input)
            logits = self.classifier(dec_out)

            logits = torch.argmax(logits, dim=-1, keepdim=True)
            last_logit = logits[-1]
            dec_logits.append(last_logit)
            dec_input = torch.cat([dec_input, last_logit], dim=-1)
        return dec_input

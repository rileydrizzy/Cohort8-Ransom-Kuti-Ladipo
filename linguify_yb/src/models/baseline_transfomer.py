"""doc
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    def __init__(self, num_vocab=1000, maxlen=100, num_hid=64):
        super(TokenEmbedding, self).__init__()
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
        super(LandmarkEmbedding, self).__init__()
        self.conv1 = nn.Conv1d(num_hid, 11, stride=2, padding="same")
        self.conv2 = nn.Conv1d(num_hid, 11, stride=2, padding="same")
        self.conv3 = nn.Conv1d(num_hid, 11, stride=2, padding="same")
        self.pos_emb = nn.Embedding(maxlen, num_hid)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return F.relu(self.conv3(x))


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        super(TransformerEncoder, self).__init__()
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

    def forward(self, inputs, training):
        attn_out, _ = self.att(inputs, inputs, inputs)
        attn_out = self.dropout1(attn_out)
        out1 = self.layernorm1(inputs + attn_out)

        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out)
        return self.layernorm2(out1 + ffn_out)


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super(TransformerDecoder, self).__init__()
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

    def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
        i = torch.arange(n_dest)[:, None]
        j = torch.arange(n_src)
        m = i >= j - n_src + n_dest
        mask = m.to(dtype)
        mask = mask.view(1, n_dest, n_src)
        mult = torch.cat(
            [batch_size[..., None], torch.tensor([1, 1], dtype=torch.int32)], 0
        )
        return mask.repeat(mult)

    def forward(self, enc_out, target, training):
        input_shape = target.size()
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = self.causal_attention_mask(
            batch_size, seq_len, seq_len, torch.bool
        )

        target_att = self.self_att(target, target, target, attn_mask=causal_mask)
        target_norm = self.layernorm1(target + self.self_dropout(target_att))

        enc_out = self.enc_att(target_norm, enc_out, enc_out)
        enc_out_norm = self.layernorm2(enc_out + self.enc_dropout(enc_out))

        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out))

        return ffn_out_norm


class Transformer(nn.Module):
    def __init__(
        self,
        num_hid=64,
        num_head=2,
        num_feed_forward=128,
        source_maxlen=100,
        target_maxlen=100,
        num_layers_enc=4,
        num_layers_dec=1,
        num_classes=60,
    ):
        super(Transformer, self).__init__()
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.target_maxlen = target_maxlen
        self.num_classes = num_classes

        self.enc_input = LandmarkEmbedding(num_hid=num_hid, maxlen=source_maxlen)
        self.dec_input = TokenEmbedding(
            num_vocab=num_classes, maxlen=target_maxlen, num_hid=num_hid
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

        self.classifier = nn.Linear(num_hid, num_classes)

    def decode(self, enc_out, target, training):
        y = self.dec_input(target)
        for i in range(self.num_layers_dec):
            y = getattr(self, f"dec_layer_{i}")(enc_out, y, training)
        return y

    def forward(self, inputs, training):
        source, target = inputs
        x = self.encoder(source)
        y = self.decode(x, target, training)
        return self.classifier(y)

    def generate(self, source, target_start_token_idx):
        bs = source.size(0)
        enc = self.encoder(source)
        dec_input = (
            torch.ones((bs, 1), dtype=torch.long).to(source.device)
            * target_start_token_idx
        )
        dec_logits = []
        for i in range(self.target_maxlen - 1):
            dec_out = self.decode(enc, dec_input, training=False)
            logits = self.classifier(dec_out)
            logits = torch.argmax(logits, dim=-1, keepdim=True)
            last_logit = logits[:, -1]
            dec_logits.append(last_logit)
            dec_input = torch.cat([dec_input, last_logit], dim=-1)
        return dec_input

def build_model():
    pass

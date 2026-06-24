from .istftnet import AdainResBlk1d
from torch.nn.utils.parametrizations import weight_norm
from transformers import AlbertModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super().__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(
            self.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x):
        return self.linear_layer(x)


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class TextEncoder(nn.Module):
    def __init__(self, channels, kernel_size, depth, n_symbols, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)
        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList(
            [
                nn.Sequential(
                    weight_norm(
                        nn.Conv1d(
                            channels, channels, kernel_size=kernel_size, padding=padding
                        )
                    ),
                    LayerNorm(channels),
                    actv,
                    nn.Dropout(0.2),
                )
                for _ in range(depth)
            ]
        )
        self.lstm = nn.LSTM(
            channels, channels // 2, 1, batch_first=True, bidirectional=True
        )

    def forward(self, x, input_lengths, m):
        valid = (~m).to(dtype=self.embedding.weight.dtype).unsqueeze(1)
        x = self.embedding(x).transpose(1, 2) * valid
        for c in self.cnn:
            x = c(x) * valid
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        return x.transpose(-1, -2) * valid


class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x, s):
        x = x.transpose(-1, -2).transpose(1, -1)
        h = self.fc(s).view(s.size(0), -1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma = gamma.transpose(1, -1)
        beta = beta.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)


class ProsodyPredictor(nn.Module):
    def __init__(self, style_dim, d_hid, nlayers, max_dur=50, dropout=0.1):
        super().__init__()
        self.text_encoder = DurationEncoder(
            sty_dim=style_dim, d_model=d_hid, nlayers=nlayers, dropout=dropout
        )
        self.lstm = nn.LSTM(
            d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True
        )
        self.duration_proj = LinearNorm(d_hid, max_dur)
        self.shared = nn.LSTM(
            d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True
        )

        self.F0 = nn.ModuleList(
            [
                AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout),
                AdainResBlk1d(
                    d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout
                ),
                AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout),
            ]
        )
        self.N = nn.ModuleList(
            [
                AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout),
                AdainResBlk1d(
                    d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout
                ),
                AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout),
            ]
        )
        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

    def forward(self, texts, style, text_lengths, alignment, m):
        d = self.text_encoder(texts, style, text_lengths, m)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(d)
        duration = self.duration_proj(F.dropout(x, 0.5, training=False))
        en = d.transpose(-1, -2) @ alignment
        return duration.squeeze(-1), en

    def F0Ntrain(self, x, s):
        self.shared.flatten_parameters()
        x, _ = self.shared(x.transpose(-1, -2))
        x = x.transpose(-1, -2)

        F0 = x
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0)

        N = x
        for block in self.N:
            N = block(N, s)
        N = self.N_proj(N)

        return F0[:, 0, :], N[:, 0, :]


class DurationEncoder(nn.Module):
    def __init__(self, sty_dim, d_model, nlayers, dropout=0.1):
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(nlayers):
            self.lstms.append(
                nn.LSTM(
                    d_model + sty_dim,
                    d_model // 2,
                    1,
                    batch_first=True,
                    bidirectional=True,
                )
            )
            self.lstms.append(AdaLayerNorm(sty_dim, d_model))
        self.dropout = dropout
        self.d_model = d_model
        self.sty_dim = sty_dim

    def forward(self, x, style, text_lengths, m):
        valid = (~m).to(dtype=x.dtype).unsqueeze(1)
        style_time = style.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        x = torch.cat([x, style_time], dim=1) * valid

        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                style_time = style.unsqueeze(-1).expand(-1, -1, x.shape[-1])
                x = torch.cat([x, style_time], dim=1) * valid
            else:
                block.flatten_parameters()
                x, _ = block(x.transpose(-1, -2))
                x = (
                    F.dropout(x, p=self.dropout, training=False).transpose(-1, -2)
                    * valid
                )

        return x.transpose(-1, -2)


class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs).last_hidden_state

from typing import Literal, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from transformers import AlbertModel

from .istftnet import AdainResBlk1d
from .my_lstm import FastGraphBiLSTM

Nonlinearity: TypeAlias = Literal[
    "linear",
    "sigmoid",
    "tanh",
    "relu",
    "leaky_relu",
    "selu",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
]


class LinearNorm(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        bias=True,
        w_init_gain: Nonlinearity = "linear",
    ):
        super().__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init_gain),
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
                            channels,
                            channels,
                            kernel_size=kernel_size,
                            padding=padding,
                        )
                    ),
                    LayerNorm(channels),
                    actv,
                    nn.Dropout(0.2),
                )
                for _ in range(depth)
            ]
        )
        raw_lstm = nn.LSTM(
            channels,
            channels // 2,
            1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm = FastGraphBiLSTM(raw_lstm, max_seq_len=512)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        for c in self.cnn:
            x = c(x)

        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        return x


class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x, s):
        x = x.transpose(-1, -2).transpose(1, -1)
        h = self.fc(s)
        gamma = h[:, : self.channels].unsqueeze(1)
        beta = h[:, self.channels :].unsqueeze(1)
        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)


class ProsodyPredictor(nn.Module):
    def __init__(self, style_dim, d_hid, nlayers, max_dur=50, dropout=0.1):
        super().__init__()
        self.text_encoder = DurationEncoder(
            sty_dim=style_dim,
            d_model=d_hid,
            nlayers=nlayers,
            dropout=dropout,
        )
        raw_lstm = nn.LSTM(
            d_hid + style_dim,
            d_hid // 2,
            1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm = FastGraphBiLSTM(raw_lstm, max_seq_len=512)
        self.duration_proj = LinearNorm(d_hid, max_dur)
        raw_shared = nn.LSTM(
            d_hid + style_dim,
            d_hid // 2,
            1,
            batch_first=True,
            bidirectional=True,
        )
        # NOTE: Give this LSTM leeway in max_seq_len as text chunks can
        # have predicted spoken durations in frames that are slightly longer than
        # their numerical phoneme count.
        self.shared = FastGraphBiLSTM(raw_shared, max_seq_len=1024)

        self.F0 = nn.ModuleList(
            [
                AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout),
                AdainResBlk1d(
                    d_hid,
                    d_hid // 2,
                    style_dim,
                    upsample=True,
                    dropout_p=dropout,
                ),
                AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout),
            ]
        )
        self.N = nn.ModuleList(
            [
                AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout),
                AdainResBlk1d(
                    d_hid,
                    d_hid // 2,
                    style_dim,
                    upsample=True,
                    dropout_p=dropout,
                ),
                AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout),
            ]
        )
        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

    def forward(self, texts, style, alignment):
        d = self.text_encoder(texts, style)

        x, _ = self.lstm(d)

        duration = self.duration_proj(F.dropout(x, 0.5, training=False))
        en = d.transpose(-1, -2) @ alignment
        return duration.squeeze(-1), en

    def F0Ntrain(self, x, s):
        x = x.transpose(-1, -2)
        x, _ = self.shared(x)
        x = x.transpose(-1, -2)

        f0 = x
        for block in self.F0:
            f0 = block(f0, s)
        f0 = self.F0_proj(f0)

        noise = x
        for block in self.N:
            noise = block(noise, s)
        noise = self.N_proj(noise)

        return f0[:, 0, :], noise[:, 0, :]


class DurationEncoder(nn.Module):
    def __init__(self, sty_dim, d_model, nlayers, dropout=0.1):
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(nlayers):
            raw_lstm = nn.LSTM(
                d_model + sty_dim,
                d_model // 2,
                1,
                batch_first=True,
                bidirectional=True,
            )
            self.lstms.append(FastGraphBiLSTM(raw_lstm, max_seq_len=512))
            self.lstms.append(AdaLayerNorm(sty_dim, d_model))
        self.dropout = dropout
        self.d_model = d_model
        self.sty_dim = sty_dim

    def forward(self, x, style):
        style_time = style.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        x = torch.cat([x, style_time], dim=1)

        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                style_time = style.unsqueeze(-1).expand(-1, -1, x.shape[-1])
                x = torch.cat([x, style_time], dim=1)
            elif isinstance(block, FastGraphBiLSTM):
                x_time = x.transpose(-1, -2)
                x_time, _ = block(x_time)
                x = F.dropout(
                    x_time,
                    p=self.dropout,
                    training=False,
                ).transpose(-1, -2)

        return x.transpose(-1, -2)


class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):  # pyright: ignore[reportIncompatibleMethodOverride]
        output = super().forward(*args, **kwargs)
        if isinstance(output, tuple):
            return output[0]
        return output.last_hidden_state

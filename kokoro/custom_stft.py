from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _frozen_conv_transpose1d(weight: torch.Tensor, stride: int) -> nn.ConvTranspose1d:
    """
    Build an nn.ConvTranspose1d module whose weight is a non-persistent buffer.

    TensorRT generally handles module ConvTranspose nodes more reliably than
    functional F.conv_transpose1d calls. The kernels are deterministic STFT
    synthesis windows, not learned parameters, so keeping them as frozen buffers
    avoids checkpoint/state_dict churn.
    """
    module = nn.ConvTranspose1d(
        in_channels=int(weight.shape[0]),
        out_channels=int(weight.shape[1]),
        kernel_size=int(weight.shape[2]),
        stride=stride,
        bias=False,
    )
    del module._parameters["weight"]
    module.register_buffer("weight", weight.contiguous().clone(), persistent=False)

    # TensorRT export preparation may replace this frozen ConvTranspose1d with
    # an exact phase-decomposed Conv1d implementation. Keeping the STFT kernels
    # as module-owned buffers here avoids checkpoint/state_dict churn before
    # export preparation runs.
    return module


class CustomSTFT(nn.Module):
    """
    Real-valued STFT/iSTFT implemented with Conv1d/ConvTranspose1d.

    This module avoids complex tensors and torch.stft/torch.istft. The inverse
    path implements correct one-sided real iDFT scaling, Hann overlap-add, and
    window-sum-square normalization.

    The inverse path intentionally uses frozen nn.ConvTranspose1d modules instead
    of functional F.conv_transpose1d calls to produce a cleaner TensorRT graph.
    """

    window: torch.Tensor  # pyright: ignore[reportUninitializedInstanceVariable]
    weight_forward_real: torch.Tensor  # pyright: ignore[reportUninitializedInstanceVariable]
    weight_forward_imag: torch.Tensor  # pyright: ignore[reportUninitializedInstanceVariable]
    weight_backward_real: torch.Tensor  # pyright: ignore[reportUninitializedInstanceVariable]
    weight_backward_imag: torch.Tensor  # pyright: ignore[reportUninitializedInstanceVariable]
    weight_window_square: torch.Tensor  # pyright: ignore[reportUninitializedInstanceVariable]

    def __init__(
        self,
        filter_length=800,
        hop_length=200,
        win_length=800,
        window="hann",
        center=True,
        pad_mode="replicate",
    ):
        super().__init__()
        assert window == "hann", window

        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = filter_length
        self.center = center
        self.pad_mode = pad_mode
        self.freq_bins = self.n_fft // 2 + 1

        win = torch.hann_window(win_length, periodic=True, dtype=torch.float32)
        if win_length < self.n_fft:
            win = F.pad(win, (0, self.n_fft - win_length))
        elif win_length > self.n_fft:
            win = win[: self.n_fft]
        self.register_buffer("window", win, persistent=False)

        n = np.arange(self.n_fft, dtype=np.float64)
        k = np.arange(self.freq_bins, dtype=np.float64)
        angle = 2.0 * np.pi * np.outer(k, n) / self.n_fft

        forward_real = np.cos(angle) * win.numpy()
        forward_imag = -np.sin(angle) * win.numpy()

        self.register_buffer(
            "weight_forward_real",
            torch.from_numpy(forward_real).float().unsqueeze(1),
            persistent=False,
        )
        self.register_buffer(
            "weight_forward_imag",
            torch.from_numpy(forward_imag).float().unsqueeze(1),
            persistent=False,
        )

        scale = np.ones(self.freq_bins, dtype=np.float64)
        if self.freq_bins > 2:
            scale[1:-1] = 2.0
        if self.n_fft % 2 == 1 and self.freq_bins > 1:
            scale[1:] = 2.0

        inv_scale = scale[:, None] / self.n_fft
        inverse_real = np.cos(angle) * inv_scale * win.numpy()
        inverse_imag = -np.sin(angle) * inv_scale * win.numpy()

        weight_backward_real = torch.from_numpy(inverse_real).float().unsqueeze(1)
        weight_backward_imag = torch.from_numpy(inverse_imag).float().unsqueeze(1)
        weight_window_square = (win * win).view(1, 1, -1)

        self.register_buffer(
            "weight_backward_real", weight_backward_real, persistent=False
        )
        self.register_buffer(
            "weight_backward_imag", weight_backward_imag, persistent=False
        )
        self.register_buffer(
            "weight_window_square", weight_window_square, persistent=False
        )

        self.deconv_real = _frozen_conv_transpose1d(
            weight_backward_real, stride=self.hop_length
        )
        self.deconv_imag = _frozen_conv_transpose1d(
            weight_backward_imag, stride=self.hop_length
        )
        self.deconv_window_square = _frozen_conv_transpose1d(
            weight_window_square, stride=self.hop_length
        )

    def transform(self, waveform: torch.Tensor):
        if waveform.dim() == 2:
            x = waveform
        elif waveform.dim() == 3 and waveform.shape[1] == 1:
            x = waveform[:, 0, :]
        else:
            raise ValueError(
                f"Expected waveform [B,T] or [B,1,T], got {waveform.shape}"
            )

        if self.center:
            pad = self.n_fft // 2
            x = F.pad(x, (pad, pad), mode=self.pad_mode)

        x = x.unsqueeze(1)
        real = F.conv1d(x, self.weight_forward_real, stride=self.hop_length)
        imag = F.conv1d(x, self.weight_forward_imag, stride=self.hop_length)

        magnitude = torch.sqrt(real.square() + imag.square() + 1e-14)
        phase = torch.atan2(imag, real)
        phase = torch.where(
            (imag == 0) & (real < 0), torch.full_like(phase, torch.pi), phase
        )
        return magnitude, phase

    def inverse(
        self, magnitude: torch.Tensor, phase: torch.Tensor, length: Optional[int] = None
    ) -> torch.Tensor:
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)

        waveform = self.deconv_real(real)
        waveform = waveform + self.deconv_imag(imag)

        envelope = self.deconv_window_square(torch.ones_like(magnitude[:, :1, :]))
        waveform = waveform / torch.clamp(envelope, min=1e-8)

        if self.center:
            pad = self.n_fft // 2
            waveform = waveform[..., pad:-pad]

        if length is not None:
            waveform = waveform[..., :length]

        return waveform

    def forward(self, x: torch.Tensor):
        mag, phase = self.transform(x)
        return self.inverse(mag, phase, length=x.shape[-1])

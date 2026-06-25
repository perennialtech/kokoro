import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from kokoro.custom_stft import CustomSTFT


def init_weights(m, mean=0.0, std=0.01):
    if m.__class__.__name__.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def conv_transpose1d_output_length(
    module: nn.ConvTranspose1d,
    length: int,
) -> int:
    stride = module.stride[0]
    padding = module.padding[0]
    dilation = module.dilation[0]
    kernel_size = module.kernel_size[0]
    output_padding = module.output_padding[0]
    return (
        (int(length) - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + output_padding
        + 1
    )


class ExplicitInstanceNorm1d(nn.Module):
    """
    InstanceNorm1d implemented explicitly with reductions.

    The parameters are intentionally named weight/bias so checkpoints using
    AdaIN1d.norm.weight and AdaIN1d.norm.bias continue to load unchanged.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.num_features))
        self.bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        var = centered.square().mean(dim=-1, keepdim=True)
        x = centered * torch.rsqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)


class AdaIN1d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.num_features = int(num_features)
        self.norm = ExplicitInstanceNorm1d(self.num_features)
        self.fc = nn.Linear(style_dim, self.num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        gamma = h[:, : self.num_features].unsqueeze(-1)
        beta = h[:, self.num_features :].unsqueeze(-1)
        return (1 + gamma) * self.norm(x) + beta


class AdaINResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), style_dim=64):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.adain1 = nn.ModuleList([AdaIN1d(style_dim, channels) for _ in range(3)])
        self.adain2 = nn.ModuleList([AdaIN1d(style_dim, channels) for _ in range(3)])
        self.alpha1 = nn.ParameterList(
            [nn.Parameter(torch.ones(1, channels, 1)) for _ in range(3)]
        )
        self.alpha2 = nn.ParameterList(
            [nn.Parameter(torch.ones(1, channels, 1)) for _ in range(3)]
        )
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x, s):
        for c1, c2, n1, n2, a1, a2 in zip(
            self.convs1,
            self.convs2,
            self.adain1,
            self.adain2,
            self.alpha1,
            self.alpha2,
        ):
            xt = n1(x, s)
            xt = xt + (1 / a1) * torch.sin(a1 * xt).square()
            xt = c1(xt)
            xt = n2(xt, s)
            xt = xt + (1 / a2) * torch.sin(a2 * xt).square()
            x = c2(xt) + x
        return x


class SineGen(nn.Module):
    def __init__(
        self,
        samp_rate,
        upsample_scale,
        harmonic_num=0,
        sine_amp=0.1,
        voiced_threshold=0,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.harmonic_num = harmonic_num
        self.dim = harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.upsample_scale = upsample_scale

    def _f02uv(self, f0):
        return (f0 > self.voiced_threshold).to(torch.float32)

    def _f02sine(self, f0_values):
        rad_values = (f0_values / self.sampling_rate) % 1

        # FIXME: Review this:
        # TRT Dynamic Shape Fix: Use scale_factor instead of explicitly computing sizes
        # that map to the dynamic inference time dimension (-1)
        scale_down = 1.0 / self.upsample_scale
        scale_up = float(self.upsample_scale)

        rad = F.interpolate(
            rad_values.transpose(1, 2),
            scale_factor=scale_down,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

        phase = torch.cumsum(rad, dim=1) * 2 * torch.pi

        phase = F.interpolate(
            phase.transpose(1, 2) * self.upsample_scale,
            scale_factor=scale_up,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

        # TRT-safe truncation: Guarantees length perfectly matches expecting input
        # (Slicing an existing tensor is absolutely TRT-compliant, unlike eager allocation)
        target_len = f0_values.shape[1]
        phase = phase[:, :target_len, :]

        return torch.sin(phase)

    def forward(self, f0):
        harmonics = torch.arange(
            1,
            self.harmonic_num + 2,
            device=f0.device,
            dtype=f0.dtype,
        ).view(1, 1, -1)
        sine_waves = self._f02sine(f0 * harmonics) * self.sine_amp
        uv = self._f02uv(f0)
        noise = torch.zeros_like(sine_waves)
        return sine_waves * uv, uv, noise


class SourceModuleHnNSF(nn.Module):
    def __init__(
        self,
        sampling_rate,
        upsample_scale,
        harmonic_num=0,
        sine_amp=0.1,
        voiced_threshod=0,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.l_sin_gen = SineGen(
            sampling_rate,
            upsample_scale,
            harmonic_num,
            sine_amp,
            voiced_threshod,
        )
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x):
        sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge, torch.zeros_like(uv), uv


class Generator(nn.Module):
    def __init__(
        self,
        style_dim,
        resblock_kernel_sizes,
        upsample_rates,
        upsample_initial_channel,
        resblock_dilation_sizes,
        upsample_kernel_sizes,
        gen_istft_n_fft,
        gen_istft_hop_size,
    ):
        super().__init__()
        if not upsample_rates:
            raise ValueError("Generator requires at least one upsample rate")
        if not resblock_kernel_sizes:
            raise ValueError("Generator requires at least one residual block kernel")

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.frame_upsample_scale = math.prod(upsample_rates)

        # F0/noise are already predicted at the same temporal resolution as the
        # generator input. Decoder.decode_features upsamples the ASR frame stream
        # once, so:
        #
        #   synthesis frames:       T
        #   F0/noise frames:        2T
        #   generator input frames: 2T
        #
        # Therefore each F0 frame must expand by the full generator sample
        # upsampling factor. Dividing this by 2 makes harmonic features half as
        # long as the generator path and breaks Decoder.forward_with_har.
        self.source_upsample_scale = self.frame_upsample_scale * gen_istft_hop_size

        self.m_source = SourceModuleHnNSF(
            24000,
            self.source_upsample_scale,
            harmonic_num=8,
            voiced_threshod=10,
        )
        self.noise_convs = nn.ModuleList()
        self.noise_res = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.source_feature_channels: list[int] = []

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(AdaINResBlock1(ch, k, d, style_dim))

            c_cur = upsample_initial_channel // (2 ** (i + 1))
            self.source_feature_channels.append(c_cur)

            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1 :])
                self.noise_convs.append(
                    nn.Conv1d(
                        gen_istft_n_fft + 2,
                        c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=(stride_f0 + 1) // 2,
                    )
                )
                self.noise_res.append(AdaINResBlock1(c_cur, 7, [1, 3, 5], style_dim))
            else:
                self.noise_convs.append(
                    nn.Conv1d(gen_istft_n_fft + 2, c_cur, kernel_size=1)
                )
                self.noise_res.append(AdaINResBlock1(c_cur, 11, [1, 3, 5], style_dim))

        self.post_n_fft = gen_istft_n_fft
        final_ch = upsample_initial_channel // (2**self.num_upsamples)
        self.conv_post = weight_norm(
            nn.Conv1d(final_ch, self.post_n_fft + 2, 7, 1, padding=3)
        )
        self.stft = CustomSTFT(gen_istft_n_fft, gen_istft_hop_size, gen_istft_n_fft)
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def source_channels(self) -> list[int]:
        return list(self.source_feature_channels)

    def source_frame_lengths(self, decoder_frames: int) -> list[int]:
        """
        Return the per-generator-layer time lengths after each upsampling step.

        These are exactly the lengths expected by the harmonic/source branch at
        each residual add. TensorRT compilation uses these lengths directly as
        source-pyramid input profile dimensions instead of asking TensorRT to
        infer them from strided convolutions over the final harmonic tensor.
        """
        length = int(decoder_frames)
        lengths: list[int] = []

        for i, up in enumerate(self.ups):
            length = conv_transpose1d_output_length(up, length)
            if i == self.num_upsamples - 1:
                length += 1
            lengths.append(length)

        if not lengths:
            raise RuntimeError("Generator has no upsampling layers")

        return lengths

    def output_frame_length(self, decoder_frames: int) -> int:
        return self.source_frame_lengths(decoder_frames)[-1]

    def compute_harmonic_features(self, f0):
        scale = int(self.source_upsample_scale)

        f0 = f0.unsqueeze(1)
        weight = torch.ones(1, 1, scale, device=f0.device, dtype=f0.dtype)
        f0 = torch.nn.functional.conv_transpose1d(f0, weight, stride=scale)
        f0 = f0.transpose(1, 2)

        har_source, _, _ = self.m_source(f0)
        har_source = har_source.transpose(1, 2)

        magnitude, phase = self.stft.transform(har_source)
        return torch.cat([magnitude, phase], dim=1)

    def compute_source_pyramid(
        self,
        har: torch.Tensor,
        s: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """
        Compute the harmonic/source tensors consumed at each generator layer.

        This is intentionally factored out for TensorRT. The old TensorRT graph
        had to prove relationships such as:

            final_har_frames -> strided Conv1d -> layer_i_frames
            decoder_frames   -> ConvTranspose path -> layer_i_frames

        Those relationships are true, but TensorRT's profile shape machine can
        reject the kMIN point for small dynamic profiles. Passing this pyramid
        as explicit engine inputs removes that fragile floor/stride proof from
        the TensorRT graph.
        """
        return tuple(
            n_res(n_conv(har), s)
            for n_res, n_conv in zip(self.noise_res, self.noise_convs)
        )

    def forward_with_source_pyramid(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        source_pyramid: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        if len(source_pyramid) != self.num_upsamples:
            raise ValueError(
                "Generator source pyramid length mismatch: "
                f"got {len(source_pyramid)}, expected {self.num_upsamples}"
            )

        for i, (x_source, up) in enumerate(zip(source_pyramid, self.ups)):
            x = F.leaky_relu(x, negative_slope=0.1)
            x = up(x)
            if i == self.num_upsamples - 1:
                x = torch.cat([x[..., 1:2], x], dim=-1)

            torch._assert(
                x.shape[1] == x_source.shape[1],
                "Generator source path channel count must match upsampled decoder path",
            )
            torch._assert(
                x.shape[-1] == x_source.shape[-1],
                "Generator source path length must match upsampled decoder path",
            )
            x = x + x_source

            xs: Optional[torch.Tensor] = None

            start_idx = i * self.num_kernels
            end_idx = start_idx + self.num_kernels

            for k in range(start_idx, end_idx):
                y = self.resblocks[k](x, s)
                xs = y if xs is None else xs + y

            if xs is None:
                raise RuntimeError("Generator has no residual blocks")

            x = xs / self.num_kernels

        x = self.conv_post(F.leaky_relu(x))
        spec = torch.exp(x[:, : self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1 :, :])
        return self.stft.inverse(spec, phase)

    def forward_with_har(self, x, s, har):
        return self.forward_with_source_pyramid(
            x,
            s,
            self.compute_source_pyramid(har, s),
        )

    def forward(self, x, s, f0):
        return self.forward_with_har(x, s, self.compute_harmonic_features(f0))


class UpSample1d(nn.Module):
    def __init__(self, enabled: bool):
        super().__init__()
        self.enabled = bool(enabled)

    def forward(self, x):
        if not self.enabled:
            return x
        return F.interpolate(x, scale_factor=2.0, mode="nearest")


class ExplicitDepthwiseConvTranspose1dStride2(nn.Module):
    """
    Exact replacement for:

        nn.ConvTranspose1d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            groups=channels,
        )

    TensorRT/Cask is fragile for this depthwise transposed-convolution shape.
    This module implements the same operation with elementwise ops, slicing and
    interleaving, avoiding grouped ConvTranspose1d in the TensorRT graph.

    For each channel c and input position i:

        y[c, 2*i]     = x[c, i] * w[c, 1]
        y[c, 2*i + 1] = x[c, i] * w[c, 2] + x[c, i + 1] * w[c, 0]

    plus ConvTranspose1d bias added to every output sample.
    """

    def __init__(self, channels: int, bias: bool = True):
        super().__init__()
        self.channels = int(channels)
        self.weight = nn.Parameter(torch.empty(self.channels, 1, 3))
        self.bias = nn.Parameter(torch.empty(self.channels)) if bias else None

    @classmethod
    def from_conv(
        cls,
        conv: nn.ConvTranspose1d,
    ) -> "ExplicitDepthwiseConvTranspose1dStride2":
        if not _is_depthwise_stride2_deconv_for_explicit_export(conv):
            raise ValueError(
                "ExplicitDepthwiseConvTranspose1dStride2 can only replace "
                "depthwise ConvTranspose1d with kernel_size=3, stride=2, "
                "padding=1, output_padding=1, dilation=1."
            )

        module = cls(conv.in_channels, bias=conv.bias is not None).to(
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )

        with torch.no_grad():
            module.weight.copy_(conv.weight.detach())
            module.weight.requires_grad_(conv.weight.requires_grad)

            if conv.bias is not None and module.bias is not None:
                module.bias.copy_(conv.bias.detach())
                module.bias.requires_grad_(conv.bias.requires_grad)

        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"Expected input [B,C,T] for explicit depthwise deconv, got {x.shape}"
            )

        torch._assert(
            x.shape[1] == self.channels,
            "Explicit depthwise deconv channel count mismatch",
        )

        w = self.weight[:, 0, :]
        w_left = w[:, 0].view(1, self.channels, 1)
        w_center = w[:, 1].view(1, self.channels, 1)
        w_right = w[:, 2].view(1, self.channels, 1)

        # Broadcast scalar-per-channel weights against the full dynamic input
        # length. Keep all elementwise arithmetic at length T.
        #
        # Avoid expressions such as:
        #
        #     right[..., :-1] + left[..., 1:]
        #
        # Although both operands have length T - 1, torch.export may still emit
        # a broadcast-specialization guard excluding T - 1 == 1, which rejects a
        # valid TensorRT profile with min_frames=2.
        left = x * w_left
        even = x * w_center
        right = x * w_right

        zero_tail = torch.zeros_like(left[..., :1])
        left_shift = torch.cat([left, zero_tail], dim=-1)[..., 1:]

        odd = right + left_shift

        y = torch.stack([even, odd], dim=-1).flatten(-2)

        if self.bias is not None:
            y = y + self.bias.view(1, self.channels, 1)

        return y


def _is_depthwise_stride2_deconv_for_explicit_export(module: nn.Module) -> bool:
    return (
        isinstance(module, nn.ConvTranspose1d)
        and module.in_channels == module.out_channels
        and module.groups == module.in_channels
        and module.kernel_size == (3,)
        and module.stride == (2,)
        and module.padding == (1,)
        and module.output_padding == (1,)
        and module.dilation == (1,)
    )


class ExplicitConvTranspose1dByPhase(nn.Module):
    """
    Exact ConvTranspose1d replacement using phase-decomposed regular Conv1d.

    For a ConvTranspose1d with dilation=1,

        output_length = stride * input_length + constant
        constant = kernel_size + output_padding - stride - 2 * padding

    Kokoro's transposed convolutions all have `constant % stride == 0`, so every
    stride phase has the same dynamic length. This module computes each output
    phase with an ordinary Conv1d over the original input, then interleaves the
    phases.

    This avoids both problematic TensorRT paths seen during debugging:

      1. Native ConvTranspose1d lowering can fail in Cask with:
            isOpConsistent(convolution.get()) failed

      2. Zero-insertion rewrites create expanded dynamic tensors whose shape
         expressions can become inconsistent at TensorRT profile minima.

    The module preserves ConvTranspose1d-style tuple attributes because Kokoro's
    shape helpers inspect `kernel_size`, `stride`, `padding`, `output_padding`,
    and `dilation`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int,
        groups: int,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        weight_requires_grad: bool,
        bias_requires_grad: bool,
    ):
        super().__init__()

        if int(kernel_size) < 1:
            raise ValueError("kernel_size must be positive")
        if int(stride) < 1:
            raise ValueError("stride must be positive")
        if int(groups) < 1:
            raise ValueError("groups must be positive")
        if int(in_channels) % int(groups) != 0:
            raise ValueError("in_channels must be divisible by groups")
        if int(out_channels) % int(groups) != 0:
            raise ValueError("out_channels must be divisible by groups")
        if int(output_padding) < 0 or int(output_padding) >= int(stride):
            raise ValueError("output_padding must satisfy 0 <= output_padding < stride")
        if int(padding) < 0:
            raise ValueError("padding must be non-negative")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.groups = int(groups)

        self.kernel_size = (int(kernel_size),)
        self.stride = (int(stride),)
        self.padding = (int(padding),)
        self.output_padding = (int(output_padding),)
        self.dilation = (1,)

        self.weight = nn.Parameter(
            weight.detach().clone(),
            requires_grad=weight_requires_grad,
        )

        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(
                bias.detach().clone(),
                requires_grad=bias_requires_grad,
            )

        constant = (
            self.kernel_size[0]
            + self.output_padding[0]
            - self.stride[0]
            - (2 * self.padding[0])
        )
        if constant % self.stride[0] != 0:
            raise ValueError(
                "ExplicitConvTranspose1dByPhase requires all output phases to "
                "have the same dynamic length; expected "
                "(kernel_size + output_padding - stride - 2 * padding) % stride == 0, "
                f"got constant={constant}, stride={self.stride[0]}."
            )

        self.output_length_extra = constant // self.stride[0]
        self.phase_specs: list[tuple[int, int, int]] = []

        for phase in range(self.stride[0]):
            first_tap = (phase + self.padding[0]) % self.stride[0]
            phase_shift = (phase + self.padding[0] - first_tap) // self.stride[0]

            if first_tap >= self.kernel_size[0]:
                raise ValueError(
                    "ConvTranspose1d phase has no contributing kernel taps: "
                    f"phase={phase}, first_tap={first_tap}, "
                    f"kernel_size={self.kernel_size[0]}"
                )

            tap_count = ((self.kernel_size[0] - 1 - first_tap) // self.stride[0]) + 1
            pad_left = tap_count - 1 - phase_shift
            pad_right = phase_shift + self.output_length_extra

            if pad_left < 0 or pad_right < 0:
                raise ValueError(
                    "ConvTranspose1d phase decomposition would require negative "
                    "padding, which is not supported for this layer: "
                    f"phase={phase}, pad_left={pad_left}, pad_right={pad_right}"
                )

            self.phase_specs.append((first_tap, pad_left, pad_right))

    @classmethod
    def from_conv(cls, conv: nn.ConvTranspose1d) -> "ExplicitConvTranspose1dByPhase":
        if conv.dilation != (1,):
            raise ValueError("ExplicitConvTranspose1dByPhase only supports dilation=1")

        weight_parameter = conv._parameters.get("weight")
        bias_parameter = conv._parameters.get("bias")

        return cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size[0],
            stride=conv.stride[0],
            padding=conv.padding[0],
            output_padding=conv.output_padding[0],
            groups=conv.groups,
            weight=conv.weight,
            bias=conv.bias,
            weight_requires_grad=bool(
                isinstance(weight_parameter, nn.Parameter)
                and weight_parameter.requires_grad
            ),
            bias_requires_grad=bool(
                isinstance(bias_parameter, nn.Parameter)
                and bias_parameter.requires_grad
            ),
        )

    def _phase_conv1d_weight(self, first_tap: int) -> torch.Tensor:
        kernel_size = self.kernel_size[0]
        stride = self.stride[0]
        in_per_group = self.in_channels // self.groups
        out_per_group = self.out_channels // self.groups

        weight = self.weight.view(
            self.groups,
            in_per_group,
            out_per_group,
            kernel_size,
        )

        # ConvTranspose contributes taps in increasing kernel-index order, while
        # Conv1d performs cross-correlation. Flip the phase tap axis so the Conv1d
        # samples exactly the same input positions as ConvTranspose1d.
        phase_weight = weight[..., first_tap::stride].flip(-1)
        phase_weight = phase_weight.permute(0, 2, 1, 3)
        return phase_weight.reshape(
            self.out_channels,
            in_per_group,
            phase_weight.shape[-1],
        ).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"Expected input [B,C,T] for explicit ConvTranspose1d, got {x.shape}"
            )

        torch._assert(
            x.shape[1] == self.in_channels,
            "Explicit ConvTranspose1d input channel count mismatch",
        )

        phases: list[torch.Tensor] = []
        for first_tap, pad_left, pad_right in self.phase_specs:
            x_phase = F.pad(x, (pad_left, pad_right)) if pad_left or pad_right else x
            phases.append(
                F.conv1d(
                    x_phase,
                    self._phase_conv1d_weight(first_tap),
                    bias=None,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=self.groups,
                )
            )

        y = torch.stack(phases, dim=-1).flatten(-2)

        if self.bias is not None:
            y = y + self.bias.view(1, self.out_channels, 1)

        return y


def count_problematic_conv_transpose1d_for_tensorrt(module: nn.Module) -> int:
    """
    Count ConvTranspose1d modules remaining before TensorRT export.

    Kokoro no longer leaves any ConvTranspose1d native for TensorRT. Native
    lowering can fail in Cask for this dynamic 1D graph, while zero-insertion
    rewrites produce fragile expanded symbolic shapes. The replacement pass uses
    exact phase-decomposed Conv1d modules instead.
    """

    return sum(1 for child in module.modules() if isinstance(child, nn.ConvTranspose1d))


def replace_conv_transpose1d_for_tensorrt(module: nn.Module) -> int:
    """
    Recursively replace all ConvTranspose1d modules with exact phase-decomposed
    Conv1d equivalents.

    This removes native ConvTranspose1d from the TensorRT graph without using
    zero-insertion. The resulting graph contains ordinary Conv1d, Pad, Stack, and
    Reshape operations with simple dynamic time dimensions.
    """

    replacements = 0

    for name, child in list(module.named_children()):
        if isinstance(child, nn.ConvTranspose1d):
            try:
                replacement = ExplicitConvTranspose1dByPhase.from_conv(child)
            except ValueError as e:
                raise ValueError(
                    "Cannot prepare ConvTranspose1d for TensorRT export at "
                    f"{module.__class__.__name__}.{name}: {e}"
                ) from e

            setattr(module, name, replacement)
            replacements += 1
            continue

        replacements += replace_conv_transpose1d_for_tensorrt(child)

    return replacements


def replace_depthwise_conv_transpose1d_for_tensorrt(module: nn.Module) -> int:
    return replace_conv_transpose1d_for_tensorrt(module)


class AdainResBlk1d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        style_dim=64,
        actv=nn.LeakyReLU(0.2),
        upsample: bool = False,
        dropout_p=0.0,
    ):
        super().__init__()
        self.actv = actv
        self.upsample_enabled = bool(upsample)
        self.upsample = UpSample1d(self.upsample_enabled)
        self.learned_sc = dim_in != dim_out
        self.dropout = nn.Dropout(dropout_p)

        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)

        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

        self.pool = (
            weight_norm(
                nn.ConvTranspose1d(
                    dim_in,
                    dim_in,
                    kernel_size=3,
                    stride=2,
                    groups=dim_in,
                    padding=1,
                    output_padding=1,
                )
            )
            if self.upsample_enabled
            else nn.Identity()
        )

    def _shortcut(self, x):
        x = self.upsample(x)
        return self.conv1x1(x) if self.learned_sc else x

    def _residual(self, x, s):
        x = self.pool(self.actv(self.norm1(x, s)))
        x = self.conv1(self.dropout(x))
        x = self.conv2(self.dropout(self.actv(self.norm2(x, s))))
        return x

    def forward(self, x, s):
        return (self._residual(x, s) + self._shortcut(x)) * math.sqrt(0.5)


class Decoder(nn.Module):
    def __init__(
        self,
        dim_in,
        style_dim,
        resblock_kernel_sizes,
        upsample_rates,
        upsample_initial_channel,
        resblock_dilation_sizes,
        upsample_kernel_sizes,
        gen_istft_n_fft,
        gen_istft_hop_size,
    ):
        super().__init__()
        self.encode = AdainResBlk1d(dim_in + 2, 1024, style_dim)
        self.decode = nn.ModuleList(
            [
                AdainResBlk1d(1024 + 2 + 64, 1024, style_dim),
                AdainResBlk1d(1024 + 2 + 64, 1024, style_dim),
                AdainResBlk1d(1024 + 2 + 64, 1024, style_dim),
                AdainResBlk1d(1024 + 2 + 64, 512, style_dim, upsample=True),
            ]
        )
        self.F0_conv = weight_norm(
            nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1)
        )
        self.N_conv = weight_norm(
            nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1)
        )
        self.asr_res = nn.Sequential(weight_norm(nn.Conv1d(512, 64, kernel_size=1)))
        self.generator = Generator(
            style_dim,
            resblock_kernel_sizes,
            upsample_rates,
            upsample_initial_channel,
            resblock_dilation_sizes,
            upsample_kernel_sizes,
            gen_istft_n_fft,
            gen_istft_hop_size,
        )

    def generator_input_frame_length(self, synthesis_frames: int) -> int:
        length = int(synthesis_frames)
        for block in self.decode:
            if block.upsample_enabled:
                length *= 2
        return length

    def source_frame_lengths(self, synthesis_frames: int) -> list[int]:
        return self.generator.source_frame_lengths(
            self.generator_input_frame_length(synthesis_frames)
        )

    def source_channels(self) -> list[int]:
        return self.generator.source_channels()

    def compute_source_pyramid(
        self,
        har: torch.Tensor,
        s: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        return self.generator.compute_source_pyramid(har, s)

    def decode_features(self, asr, f0_curve, noise, s):
        f0 = self.F0_conv(f0_curve.unsqueeze(1))
        noise_features = self.N_conv(noise.unsqueeze(1))
        torch._assert(
            f0.shape[-1] == asr.shape[-1],
            "Decoder F0 features must match ASR frame length",
        )
        torch._assert(
            noise_features.shape[-1] == asr.shape[-1],
            "Decoder noise features must match ASR frame length",
        )

        x = self.encode(torch.cat([asr, f0, noise_features], dim=1), s)
        asr_res = self.asr_res(asr)

        use_res = True
        for block in self.decode:
            if use_res:
                x = torch.cat([x, asr_res, f0, noise_features], dim=1)
            x = block(x, s)
            if block.upsample_enabled:
                use_res = False
        return x

    def forward_with_source_pyramid(
        self,
        asr,
        f0_curve,
        noise,
        s,
        source_pyramid: tuple[torch.Tensor, ...],
    ):
        return self.generator.forward_with_source_pyramid(
            self.decode_features(asr, f0_curve, noise, s),
            s,
            source_pyramid,
        )

    def forward_with_har(self, asr, f0_curve, noise, s, har):
        return self.forward_with_source_pyramid(
            asr,
            f0_curve,
            noise,
            s,
            self.compute_source_pyramid(har, s),
        )

    def forward(self, asr, f0_curve, noise, s):
        return self.generator(
            self.decode_features(asr, f0_curve, noise, s),
            s,
            f0_curve,
        )

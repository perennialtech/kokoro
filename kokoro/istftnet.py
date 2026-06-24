from kokoro.custom_stft import CustomSTFT
from torch.nn.utils.parametrizations import weight_norm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m, mean=0.0, std=0.01):
    if m.__class__.__name__.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class AdaIN1d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=True)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s).view(s.size(0), -1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdaINResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), style_dim=64):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=get_padding(kernel_size, dilation[2]))),
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
        ])
        self.adain1 = nn.ModuleList([AdaIN1d(style_dim, channels) for _ in range(3)])
        self.adain2 = nn.ModuleList([AdaIN1d(style_dim, channels) for _ in range(3)])
        self.alpha1 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for _ in range(3)])
        self.alpha2 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for _ in range(3)])
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x, s):
        for c1, c2, n1, n2, a1, a2 in zip(self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2):
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
        noise_std=0.003,
        voiced_threshold=0,
        flag_for_pulse=False,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        self.upsample_scale = upsample_scale

    def _f02uv(self, f0):
        return (f0 > self.voiced_threshold).to(torch.float32)

    def _f02sine(self, f0_values):
        rad_values = (f0_values / self.sampling_rate) % 1

        if not self.flag_for_pulse:
            down_len = max(1, rad_values.shape[1] // self.upsample_scale)
            rad = F.interpolate(rad_values.transpose(1, 2), size=down_len, mode="linear", align_corners=False).transpose(1, 2)
            phase = torch.cumsum(rad, dim=1) * 2 * torch.pi
            phase = F.interpolate(
                (phase.transpose(1, 2) * self.upsample_scale),
                size=f0_values.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            return torch.sin(phase)

        uv = self._f02uv(f0_values)
        uv_1 = torch.roll(uv, shifts=-1, dims=1)
        uv_1[:, -1, :] = 1
        u_loc = (uv < 1) * (uv_1 > 0)
        tmp_cumsum = torch.cumsum(rad_values, dim=1)
        for idx in range(f0_values.shape[0]):
            temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
            temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[:-1, :]
            tmp_cumsum[idx, :, :] = 0
            tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum
        return torch.cos(torch.cumsum(rad_values - tmp_cumsum, dim=1) * 2 * torch.pi)

    def forward(self, f0):
        harmonics = torch.arange(1, self.harmonic_num + 2, device=f0.device, dtype=f0.dtype).view(1, 1, -1)
        sine_waves = self._f02sine(f0 * harmonics) * self.sine_amp
        uv = self._f02uv(f0)
        noise = torch.zeros_like(sine_waves)
        return sine_waves * uv, uv, noise


class SourceModuleHnNSF(nn.Module):
    def __init__(self, sampling_rate, upsample_scale, harmonic_num=0, sine_amp=0.1, add_noise_std=0.003, voiced_threshod=0):
        super().__init__()
        self.sine_amp = sine_amp
        self.l_sin_gen = SineGen(sampling_rate, upsample_scale, harmonic_num, sine_amp, add_noise_std, voiced_threshod)
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
        disable_complex=False,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.source_upsample_scale = math.prod(upsample_rates) * gen_istft_hop_size

        self.m_source = SourceModuleHnNSF(24000, self.source_upsample_scale, harmonic_num=8, voiced_threshod=10)
        self.noise_convs = nn.ModuleList()
        self.noise_res = nn.ModuleList()
        self.ups = nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(nn.ConvTranspose1d(
                upsample_initial_channel // (2 ** i),
                upsample_initial_channel // (2 ** (i + 1)),
                k,
                u,
                padding=(k - u) // 2,
            )))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(AdaINResBlock1(ch, k, d, style_dim))

            c_cur = upsample_initial_channel // (2 ** (i + 1))
            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1:])
                self.noise_convs.append(nn.Conv1d(
                    gen_istft_n_fft + 2,
                    c_cur,
                    kernel_size=stride_f0 * 2,
                    stride=stride_f0,
                    padding=(stride_f0 + 1) // 2,
                ))
                self.noise_res.append(AdaINResBlock1(c_cur, 7, [1, 3, 5], style_dim))
            else:
                self.noise_convs.append(nn.Conv1d(gen_istft_n_fft + 2, c_cur, kernel_size=1))
                self.noise_res.append(AdaINResBlock1(c_cur, 11, [1, 3, 5], style_dim))

        self.post_n_fft = gen_istft_n_fft
        self.conv_post = weight_norm(nn.Conv1d(ch, self.post_n_fft + 2, 7, 1, padding=3))
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.stft = CustomSTFT(gen_istft_n_fft, gen_istft_hop_size, gen_istft_n_fft)
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    @torch.no_grad()
    def compute_harmonic_features(self, f0):
        target = f0.shape[-1] * self.source_upsample_scale
        f0 = F.interpolate(f0[:, None], size=target, mode="nearest").transpose(1, 2)
        har_source, _, _ = self.m_source(f0)
        har_source = har_source.transpose(1, 2)
        har_spec, har_phase = self.stft.transform(har_source)
        return torch.cat([har_spec, har_phase], dim=1)

    def forward_with_har(self, x, s, har):
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, negative_slope=0.1)
            x_source = self.noise_res[i](self.noise_convs[i](har), s)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            x = x + x_source

            xs = None
            for j in range(self.num_kernels):
                y = self.resblocks[i * self.num_kernels + j](x, s)
                xs = y if xs is None else xs + y
            x = xs / self.num_kernels

        x = self.conv_post(F.leaky_relu(x))
        spec = torch.exp(x[:, : self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1 :, :])
        return self.stft.inverse(spec, phase)

    def forward(self, x, s, f0):
        return self.forward_with_har(x, s, self.compute_harmonic_features(f0))


class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == "none":
            return x
        return torch.repeat_interleave(x, repeats=2, dim=-1)


class AdainResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, actv=nn.LeakyReLU(0.2), upsample="none", dropout_p=0.0):
        super().__init__()
        self.actv = actv
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self.dropout = nn.Dropout(dropout_p)

        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)

        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

        self.pool = nn.Identity() if upsample == "none" else weight_norm(
            nn.ConvTranspose1d(dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1, output_padding=1)
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
        dim_out,
        resblock_kernel_sizes,
        upsample_rates,
        upsample_initial_channel,
        resblock_dilation_sizes,
        upsample_kernel_sizes,
        gen_istft_n_fft,
        gen_istft_hop_size,
        disable_complex=False,
    ):
        super().__init__()
        self.encode = AdainResBlk1d(dim_in + 2, 1024, style_dim)
        self.decode = nn.ModuleList([
            AdainResBlk1d(1024 + 2 + 64, 1024, style_dim),
            AdainResBlk1d(1024 + 2 + 64, 1024, style_dim),
            AdainResBlk1d(1024 + 2 + 64, 1024, style_dim),
            AdainResBlk1d(1024 + 2 + 64, 512, style_dim, upsample=True),
        ])
        self.F0_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        self.N_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
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
            disable_complex=disable_complex,
        )

    def decode_features(self, asr, F0_curve, N, s):
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N = self.N_conv(N.unsqueeze(1))
        x = self.encode(torch.cat([asr, F0, N], dim=1), s)
        asr_res = self.asr_res(asr)

        use_res = True
        for block in self.decode:
            if use_res:
                x = torch.cat([x, asr_res, F0, N], dim=1)
            x = block(x, s)
            if block.upsample_type != "none":
                use_res = False
        return x

    def forward_with_har(self, asr, F0_curve, N, s, har):
        return self.generator.forward_with_har(self.decode_features(asr, F0_curve, N, s), s, har)

    def forward(self, asr, F0_curve, N, s):
        return self.generator(self.decode_features(asr, F0_curve, N, s), s, F0_curve)

import torch
import torch.nn.functional as F

from kokoro.istftnet import Generator, SineGen


def _legacy_sinegen_forward(
    sine_gen: SineGen,
    low_rate_f0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = sine_gen.upsample_scale
    expanded_f0 = F.conv_transpose1d(
        low_rate_f0.unsqueeze(1),
        torch.ones(
            1,
            1,
            scale,
            dtype=low_rate_f0.dtype,
            device=low_rate_f0.device,
        ),
        stride=scale,
    ).transpose(1, 2)

    harmonics = torch.arange(
        1,
        sine_gen.harmonic_num + 2,
        dtype=low_rate_f0.dtype,
        device=low_rate_f0.device,
    ).view(1, 1, -1)
    f0_values = expanded_f0 * harmonics
    rad_values = (f0_values / sine_gen.sampling_rate) % 1
    rad = F.interpolate(
        rad_values.transpose(1, 2),
        scale_factor=1.0 / scale,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)
    phase = torch.cumsum(rad, dim=1) * 2 * torch.pi
    phase = F.interpolate(
        phase.transpose(1, 2) * scale,
        scale_factor=float(scale),
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)
    phase = phase[:, : expanded_f0.shape[1], :]

    sine_waves = torch.sin(phase) * sine_gen.sine_amp
    uv = sine_gen._f02uv(expanded_f0)
    noise = torch.zeros_like(sine_waves)
    return sine_waves * uv, uv, noise


def _legacy_harmonic_features(
    generator: Generator,
    low_rate_f0: torch.Tensor,
) -> torch.Tensor:
    sine_waves, _, _ = _legacy_sinegen_forward(
        generator.m_source.l_sin_gen,
        low_rate_f0,
    )
    har_source = generator.m_source.l_tanh(generator.m_source.l_linear(sine_waves))
    magnitude, phase = generator.stft.transform(har_source.transpose(1, 2))
    return torch.cat([magnitude, phase], dim=1)


def test_sinegen_low_rate_f0_matches_legacy_expanded_path():
    sine_gen = SineGen(
        samp_rate=24000,
        upsample_scale=8,
        harmonic_num=3,
        sine_amp=0.1,
        voiced_threshold=10,
    )
    low_rate_f0 = torch.tensor(
        [[0.0, 10.0, 10.25, 125.0, 220.0, 0.0, 440.0]],
        dtype=torch.float32,
    )

    actual = sine_gen(low_rate_f0.unsqueeze(-1))
    expected = _legacy_sinegen_forward(sine_gen, low_rate_f0)

    for actual_tensor, expected_tensor in zip(actual, expected):
        assert actual_tensor.shape == expected_tensor.shape
        assert torch.allclose(actual_tensor, expected_tensor, rtol=1e-6, atol=1e-6)

    source_frames = low_rate_f0.shape[1] * sine_gen.upsample_scale
    assert actual[0].shape == (1, source_frames, sine_gen.harmonic_num + 1)
    assert actual[1].shape == (1, source_frames, 1)
    assert actual[2].shape == (1, source_frames, sine_gen.harmonic_num + 1)


def test_generator_harmonic_features_match_legacy_expanded_path():
    generator = Generator(
        style_dim=4,
        resblock_kernel_sizes=[3],
        upsample_rates=[2],
        upsample_initial_channel=4,
        resblock_dilation_sizes=[[1, 3, 5]],
        upsample_kernel_sizes=[4],
        gen_istft_n_fft=8,
        gen_istft_hop_size=2,
    ).eval()
    low_rate_f0 = torch.tensor(
        [[0.0, 10.0, 10.25, 125.0, 220.0, 0.0, 440.0]],
        dtype=torch.float32,
    )

    with torch.inference_mode():
        actual = generator.compute_harmonic_features(low_rate_f0)
        expected = _legacy_harmonic_features(generator, low_rate_f0)

    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert actual.shape[-1] == expected.shape[-1]

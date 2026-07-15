import os

import pytest
import torch

from kokoro import KokoroTRT
from kokoro.telemetry import InMemoryTraceSink, ProfilerConfig, Telemetry


def artifact_dir() -> str:
    value = os.getenv("KOKORO_TRT_ARTIFACT_DIR")
    if not value:
        pytest.skip("KOKORO_TRT_ARTIFACT_DIR is required for TensorRT runtime tests")
    return value


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="TensorRT runtime requires CUDA"
)
@pytest.mark.parametrize("telemetry_enabled", [False, True])
def test_trt_runtime_synthesizes_inside_profile(telemetry_enabled):
    telemetry = (
        Telemetry(ProfilerConfig(enabled=True), [InMemoryTraceSink()])
        if telemetry_enabled
        else None
    )
    tts = KokoroTRT(
        artifact_dir(),
        verify_internal_shapes=True,
        telemetry=telemetry,
    )

    voice = os.getenv("KOKORO_TRT_VOICE", "af_heart")
    language = os.getenv("KOKORO_TRT_LANG", voice[0])

    results = list(
        tts.synthesize(
            text="Hello from Kokoro TensorRT.",
            voice=voice,
            language=language,
            speed=1.0,
        )
    )

    assert results
    result = results[0]
    assert result.audio.numel() > 0
    assert result.sample_length > 0

    generator_frames = tts.shape_plan.generator_frames(
        tts.host.model,
        result.synthesis_frame_length,
    )
    assert (
        tts.metadata.shapes["lower"]["x"][-1]
        <= generator_frames
        <= tts.metadata.shapes["upper"]["x"][-1]
    )

    if telemetry_enabled:
        assert result.profile is not None
        assert tts.telemetry.last_request_trace is not None
        assert tts.telemetry.last_request_trace.status == "ok"

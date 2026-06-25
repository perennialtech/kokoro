import os

import pytest
import torch

from kokoro import KokoroTRT
from kokoro.types import FrameItem


def artifact_dir() -> str:
    value = os.getenv("KOKORO_TRT_ARTIFACT_DIR")
    if not value:
        pytest.skip("KOKORO_TRT_ARTIFACT_DIR is required for TensorRT runtime tests")
    return value


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="TensorRT runtime requires CUDA"
)
def test_trt_runtime_synthesizes_inside_profile():
    tts = KokoroTRT(artifact_dir())

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
    assert (
        tts.min_synthesis_frames
        <= result.synthesis_frame_length
        <= tts.max_synthesis_frames
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="TensorRT runtime requires CUDA"
)
def test_trt_runtime_out_of_profile_raises():
    tts = KokoroTRT(artifact_dir())

    too_many_frames = tts.max_synthesis_frames + 1
    frame_item = FrameItem(
        asr=torch.zeros(1, too_many_frames, device="cuda"),
        en=torch.zeros(1, too_many_frames, device="cuda"),
        pred_dur=torch.ones(1, dtype=torch.long, device="cuda"),
        synthesis_frame_length=too_many_frames,
        return_frame_length=too_many_frames,
    )
    ref_s = torch.zeros(1, 256, device="cuda")

    with pytest.raises(RuntimeError, match="outside the TensorRT profile"):
        tts.render_frame(frame_item, ref_s)

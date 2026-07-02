from unittest.mock import MagicMock

import pytest
import torch

from kokoro.runtime import synthesize_prepared_trt


class DummyToken:
    def __init__(self, phonemes, whitespace):
        self.phonemes = phonemes
        self.whitespace = whitespace
        self.start_ts = None
        self.end_ts = None


@pytest.fixture
def mock_tts(monkeypatch):
    tts = MagicMock()
    tts.device = "cpu"

    chunk = MagicMock()
    chunk.trace = MagicMock()
    chunk.span.return_value.__enter__ = MagicMock()
    chunk.span.return_value.__exit__ = MagicMock()
    tts.telemetry.start_chunk.return_value = chunk
    tts.telemetry.start_chunk.owner = True

    def mock_text_duration(input_ids, ref_s, speed):
        n = input_ids.shape[1]
        df = torch.ones((1, n), dtype=torch.float32)
        dh = torch.zeros((1, n, 256), dtype=torch.float32)
        th = torch.zeros((1, n, 256), dtype=torch.float32)
        return df, dh, th

    tts.host.text_duration = mock_text_duration

    def mock_render_frame(frame_item, ref_s, profile=None):
        # Return valid mock audio depending on synthesis length
        return torch.ones((1, 1, frame_item.synthesis_frame_length * 240))

    tts.render_frame = mock_render_frame

    # Patch expand_frames to aggressively catch any invalid reading of the EOS frame.
    from kokoro.runtime import expand_frames

    original_expand_frames = expand_frames

    def mocked_expand_frames(*args, **kwargs):
        item = original_expand_frames(*args, **kwargs)
        # Inflate EOS duration massively so that if it is read as space_dur,
        # it spikes the output end_ts drastically.
        item.pred_dur[-1] = 99999
        return item

    monkeypatch.setattr("kokoro.runtime.expand_frames", mocked_expand_frames)

    return tts


@pytest.mark.parametrize(
    "text,token_configs,dur_len",
    [
        ("right > <", [("rIt", True), ("", True), ("", False)], 4),
        ("hello >~< world", [("h@lO", True), ("", True), ("w3ld", False)], 6),
        (">~< hello", [("", True), ("h@lO", False)], 4),
        ("hello >~<", [("h@lO", True), ("", False)], 3),
        ("hello >~< world > <", [("a" * 20, True), ("b", True), ("", False)], 5),
    ],
)
def test_synthesis_metadata_resilience(mock_tts, text, token_configs, dur_len):
    prepared = MagicMock()
    prepared.input_ids = torch.zeros((1, dur_len), dtype=torch.long)
    prepared.ref_s = torch.zeros((1, 256), dtype=torch.float32)
    prepared.speed = torch.ones((1,), dtype=torch.float32)
    prepared.input_length = dur_len

    prepared.tokens = [DummyToken(p, w) for p, w in token_configs]

    try:
        # timestamp metadata is best-effort and never aborts inference
        result = synthesize_prepared_trt(mock_tts, prepared)
    except IndexError as e:
        pytest.fail(f"timestamp logic threw IndexError for input {text!r}: {e}")
    except Exception as e:
        pytest.fail(f"inference aborted unexpectedly for input {text!r}: {e}")

    # synthesis still returns audio
    assert result.audio is not None, f"Expected audio output for {text!r}"

    # no read of EOS as space_dur
    for t in prepared.tokens:
        if t.start_ts is not None and t.end_ts is not None:
            duration = t.end_ts - t.start_ts
            assert duration < 500.0, (
                f"Token mistakenly read EOS padding for calculating space_dur! {text!r}"
            )

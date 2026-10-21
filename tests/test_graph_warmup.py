from unittest.mock import MagicMock

import pytest
import torch

import kokoro.trt as trt
from kokoro import KokoroTRT


def empty_runtime():
    runtime = KokoroTRT.__new__(KokoroTRT)
    runtime._inference_stream = MagicMock()
    runtime._graph_math_mode = None
    runtime._text_graph_max_tokens = 0
    runtime._render_graph_max_frames = 0
    return runtime


def test_warmup_is_cached_until_math_settings_change(monkeypatch):
    runtime = empty_runtime()
    runtime._ensure_graphs()
    runtime._ensure_graphs()
    runtime._inference_stream.synchronize.assert_called_once()
    monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: "changed")
    runtime._ensure_graphs()
    assert runtime._inference_stream.synchronize.call_count == 2


def test_failed_warmup_does_not_keep_a_valid_signature(monkeypatch):
    runtime = empty_runtime()
    runtime.host = MagicMock()
    runtime._text_graph_max_tokens = 3
    monkeypatch.setattr(
        trt, "TextDurationGraphs", MagicMock(side_effect=RuntimeError("capture failed"))
    )
    with pytest.raises(RuntimeError, match="capture failed"):
        runtime._ensure_graphs()
    assert runtime._graph_math_mode is None
    runtime._text_graph_max_tokens = 0
    runtime._ensure_graphs()
    assert runtime._graph_math_mode is not None

from unittest.mock import MagicMock

import pytest
import torch

import kokoro.trt as trt
from kokoro import KokoroTRT
from kokoro.graphs import RenderGraphs, TextDurationGraphs


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


def test_text_graph_borrows_hidden_buffers_but_owns_durations():
    graphs = TextDurationGraphs.__new__(TextDurationGraphs)
    graphs.input_ids = torch.zeros((1, 3), dtype=torch.long)
    graphs.ref_s = torch.zeros((1, 256))
    graphs.speed = torch.ones(1)
    outputs = (torch.ones(1, 3), torch.ones(1, 3, 8), torch.ones(1, 3, 4))
    replay = MagicMock()
    graphs.graphs = {3: (replay, outputs)}

    actual = graphs(
        torch.ones(1, 3, dtype=torch.long), torch.ones(1, 256), torch.ones(1)
    )

    replay.replay.assert_called_once()
    assert actual[0].data_ptr() != outputs[0].data_ptr()
    assert actual[1].data_ptr() == outputs[1].data_ptr()
    assert actual[2].data_ptr() == outputs[2].data_ptr()


def test_render_graph_borrows_output_slots():
    graphs = RenderGraphs.__new__(RenderGraphs)
    graphs.asr = torch.zeros((1, 4, 3))
    graphs.en = torch.zeros((1, 8, 3))
    graphs.ref_s = torch.zeros((1, 256))
    graphs.outputs = (torch.ones(1, 2, 5), torch.ones(1, 3, 7))
    replay = MagicMock()
    graphs.graphs = {3: (replay, (3, 6))}

    actual = graphs(
        torch.ones_like(graphs.asr),
        torch.ones_like(graphs.en),
        torch.ones_like(graphs.ref_s),
    )

    replay.replay.assert_called_once()
    assert [value.shape[-1] for value in actual] == [3, 6]
    assert all(
        value.untyped_storage().data_ptr() == slot.untyped_storage().data_ptr()
        for value, slot in zip(actual, graphs.outputs)
    )

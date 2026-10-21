import os
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from kokoro import KokoroTRT
from kokoro.benchmark import SHORT_TEXTS


@pytest.fixture(scope="module")
def runtime():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    artifact = os.getenv("KOKORO_TRT_ARTIFACT_DIR")
    if not artifact:
        pytest.skip("requires KOKORO_TRT_ARTIFACT_DIR")
    original = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        result = KokoroTRT(artifact, verify_internal_shapes=True)
        result.warmup()
        yield result
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original


@torch.inference_mode()
@pytest.mark.parametrize("length", [3, 8, 17, 33, 64, 96, 97])
def test_text_graphs_match_eager_and_own_outputs(runtime, length):
    assert not runtime.host.training
    torch.manual_seed(length)
    ids = torch.randint(0, 100, (1, length), device="cuda")
    style = torch.randn(1, 256, device="cuda")
    speed = torch.ones(1, device="cuda")
    expected = tuple(x.clone() for x in runtime.host.text_duration(ids, style, speed))
    actual = runtime._text_duration(ids, style, speed)
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output, reference, rtol=5e-5, atol=5e-6)
    saved = tuple(x.clone() for x in actual)
    changed = runtime._text_duration(ids.flip(1), style + 0.1, speed * 1.3)
    for output, reference in zip(actual, saved):
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
    assert not torch.allclose(changed[0], actual[0])


@torch.inference_mode()
@pytest.mark.parametrize("text", SHORT_TEXTS)
def test_graph_runtime_waveform_matches_eager(runtime, text):
    prepared = next(runtime.prepare(text, "af_heart", "zh"))
    text_graphs, render_graphs = runtime._text_graphs, runtime._render_graphs
    try:
        runtime._text_graphs = runtime._render_graphs = None
        reference = runtime.synthesize_prepared(prepared)
        runtime._text_graphs, runtime._render_graphs = text_graphs, render_graphs
        actual = runtime.synthesize_prepared(prepared)
        torch.testing.assert_close(reference.pred_dur, actual.pred_dur, rtol=0, atol=0)
        assert reference.sample_length == actual.sample_length
        torch.testing.assert_close(actual.audio, reference.audio, rtol=0, atol=0)
    finally:
        runtime._text_graphs, runtime._render_graphs = text_graphs, render_graphs


@torch.inference_mode()
def test_graph_replay_uses_current_speed(runtime):
    prepared = next(runtime.prepare("Please try again.", "af_heart", "zh"))
    normal = runtime.synthesize_prepared(prepared)
    prepared.speed = prepared.speed * 1.3
    faster = runtime.synthesize_prepared(prepared)
    torch.testing.assert_close(faster.duration_float, normal.duration_float / 1.3)
    assert faster.sample_length < normal.sample_length


def test_synthesis_is_safe_across_threads_and_streams(runtime):
    inputs = [
        next(runtime.prepare(text, "af_heart", "zh"))
        for text in ("Hi.", "Hello, world!", "Please try again.")
    ]
    expected = [runtime.synthesize_prepared(item).audio.cpu() for item in inputs]

    def synthesize(index):
        with torch.cuda.stream(torch.cuda.Stream()):
            return runtime.synthesize_prepared(inputs[index]).audio.cpu()

    with ThreadPoolExecutor(max_workers=3) as executor:
        actual = list(executor.map(synthesize, range(len(inputs))))
    for waveform, reference in zip(actual, expected):
        torch.testing.assert_close(waveform, reference, rtol=1e-4, atol=1e-5)


@torch.inference_mode()
def test_math_mode_change_rebuilds_graphs(runtime):
    original = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = not original
        runtime.warmup()
        assert runtime._graph_math_mode[0] == (not original)
        ids = torch.tensor([[0, 43, 0]], device="cuda")
        style, speed = torch.zeros(1, 256, device="cuda"), torch.ones(1, device="cuda")
        expected = tuple(
            x.clone() for x in runtime.host.text_duration(ids, style, speed)
        )
        actual = runtime._text_duration(ids, style, speed)
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original
        runtime.warmup()

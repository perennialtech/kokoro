"""Keep the default engine build choice tied to measured GPU performance."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

from kokoro import KokoroTRT
from kokoro import compile as compiler
from kokoro.trt_builder import build_engine_from_onnx


def test_cli_uses_optimized_builder_by_default(monkeypatch, tmp_path):
    requests = []
    monkeypatch.setattr(compiler, "compile_artifact", lambda **kw: requests.append(kw))
    monkeypatch.setattr(sys, "argv", ["kokoro.compile", "--output-dir", str(tmp_path)])
    compiler.main()
    assert requests[0]["builder_optimization_level"] == 3


def test_cli_allows_explicit_builder_opt_out(monkeypatch, tmp_path):
    requests = []
    monkeypatch.setattr(compiler, "compile_artifact", lambda **kw: requests.append(kw))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kokoro.compile",
            "--output-dir",
            str(tmp_path),
            "--builder-optimization-level",
            "0",
        ],
    )
    compiler.main()
    assert requests[0]["builder_optimization_level"] == 0


def test_graph_capture_default_covers_measured_short_input_frame_lengths():
    import inspect

    assert (
        inspect.signature(KokoroTRT).parameters["render_graph_max_frames"].default
        == 160
    )


def test_fp16_engine_uses_exported_precision_without_builder_flag(
    monkeypatch, tmp_path
):
    builder = MagicMock()
    builder.build_serialized_network.return_value = b"engine"
    parser = MagicMock()
    parser.parse_from_file.return_value = True
    trt = SimpleNamespace(
        Logger=MagicMock(WARNING=2),
        Builder=MagicMock(return_value=builder),
        OnnxParser=MagicMock(return_value=parser),
        BuilderFlag=SimpleNamespace(),
    )
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    output = tmp_path / "engine.plan"

    build_engine_from_onnx(
        onnx_path=tmp_path / "generator.onnx",
        engine_path=output,
        shapes={
            "lower": {"x": (1, 2, 4)},
            "preferred": {"x": (1, 2, 8)},
            "upper": {"x": (1, 2, 16)},
        },
        input_order=("x",),
        precision="fp16",
        workspace_size=None,
        builder_optimization_level=None,
    )

    assert output.read_bytes() == b"engine"
    builder.create_builder_config.return_value.set_flag.assert_not_called()

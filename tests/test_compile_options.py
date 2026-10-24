"""Keep the default engine build choice tied to measured GPU performance."""

import sys

from kokoro import compile as compiler
from kokoro import KokoroTRT


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

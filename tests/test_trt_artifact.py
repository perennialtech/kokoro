import json
import os
from pathlib import Path

import pytest

from kokoro.config import (CONFIG_FILENAME, HOST_STATE_FILENAME,
                           TRT_METADATA_FILENAME)
from kokoro.trt import TRT_ENGINE_FILENAME


def artifact_dir() -> Path:
    value = os.getenv("KOKORO_TRT_ARTIFACT_DIR")
    if not value:
        pytest.skip("KOKORO_TRT_ARTIFACT_DIR is required for TensorRT artifact tests")
    return Path(value)


def test_compiler_artifact_contains_required_files():
    root = artifact_dir()

    assert (root / CONFIG_FILENAME).is_file()
    assert (root / HOST_STATE_FILENAME).is_file()
    assert (root / TRT_METADATA_FILENAME).is_file()
    assert (root / TRT_ENGINE_FILENAME).is_file()


def test_trt_metadata_precision_profile_and_source_shapes():
    root = artifact_dir()

    metadata = json.loads((root / TRT_METADATA_FILENAME).read_text())

    assert metadata["artifact_type"] == "kokoro_generator_with_source_pyramid_tensorrt"
    assert metadata["engine_file"] == TRT_ENGINE_FILENAME
    assert metadata["config_file"] == CONFIG_FILENAME
    assert metadata["host_state_file"] == HOST_STATE_FILENAME
    assert metadata["precision"] in {"fp32", "fp16"}

    profile = metadata["profile"]
    assert profile["min_frames"] >= 1
    assert profile["opt_frames"] >= profile["min_frames"]
    assert profile["max_frames"] >= profile["opt_frames"]

    shapes = metadata["shapes"]
    assert set(shapes) == {"min", "opt", "max"}

    for group in ("min", "opt", "max"):
        assert "x" in shapes[group]
        assert "ref_s" in shapes[group]

        source_names = sorted(
            name for name in shapes[group] if name.startswith("source_")
        )
        assert source_names

        assert shapes[group]["x"][0] == 1
        assert shapes[group]["ref_s"] == [1, 256]

        for name in source_names:
            shape = shapes[group][name]
            assert len(shape) == 3
            assert shape[0] == 1
            assert shape[1] > 0
            assert shape[2] > 0

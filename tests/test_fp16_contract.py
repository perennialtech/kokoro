from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from kokoro.artifact import ArtifactMetadata
from kokoro.istftnet import ExplicitInstanceNorm1d
from kokoro.native_trt import NativeTRTEngine


def test_fp16_only_artifact_rejects_fp32():
    shapes = {
        key: {"x": (1, 2, 4), "ref_s": (1, 256), "source_0": (1, 2, 8)}
        for key in ("lower", "preferred", "upper")
    }
    metadata = ArtifactMetadata.create(
        repo_id="hexgrad/Kokoro-82M",
        checkpoint={},
        gpu={"compute_capability": "sm_86"},
        versions={},
        workspace_size=None,
        builder_optimization_level=None,
        shapes=shapes,
        input_names=("x", "ref_s", "source_0"),
        output_names=("audio",),
        onnx_opset=18,
        tensorrt_runtime_api={"execute": "execute_async_v3"},
    )
    assert metadata.precision == "fp16"
    with pytest.raises(ValueError, match="Only FP16"):
        replace(metadata, precision="fp32").validate()


def test_half_instance_norm_uses_float32_moments():
    x = torch.tensor([[[512.0, -512.0, 256.0, -256.0]]], dtype=torch.float16)
    actual = ExplicitInstanceNorm1d(1)(x)
    expected = ExplicitInstanceNorm1d(1)(x.float()).half()
    assert actual.dtype == torch.float16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_native_engine_rejects_wrong_tensor_dtype_before_dispatch():
    engine = NativeTRTEngine.__new__(NativeTRTEngine)
    engine.input_names = ("x",)
    engine.output_names = ("audio",)
    engine.tensor_dtypes = {"x": torch.float16}
    tensor = SimpleNamespace(
        is_cuda=True, is_contiguous=lambda: True, dtype=torch.float32
    )
    with pytest.raises(TypeError, match="requires torch.float16"):
        engine.run({"x": tensor})

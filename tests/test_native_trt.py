import os
from pathlib import Path

import pytest
import torch

from kokoro import compile_artifact
from kokoro.artifact import TensorRTArtifact
from kokoro.model import GeneratorExportBuilder, KokoroModelLoader
from kokoro.native_trt import NativeTRTEngine
from kokoro.shapes import ShapePlan
from kokoro.telemetry import InMemoryTraceSink, ProfilerConfig, Telemetry
from kokoro.trt_builder import build_engine_from_onnx


def existing_artifact_dir() -> Path:
    value = os.getenv("KOKORO_TRT_ARTIFACT_DIR")
    if not value:
        pytest.skip("KOKORO_TRT_ARTIFACT_DIR is required")
    return Path(value)


def load_model_for_artifact(artifact: TensorRTArtifact):
    loader = KokoroModelLoader(
        repo_id=artifact.metadata.repo_id,
        config=artifact.load_config(),
        model=None,
    )
    model = loader.load(load_weights=False)
    model.load_host_state(artifact.paths.host_state_path)
    return model


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="native TensorRT tests require CUDA"
)
@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_native_compile_builds_artifact_and_metadata_loads(tmp_path, precision):
    if os.getenv("KOKORO_NATIVE_TRT_BUILD_TESTS") != "1":
        pytest.skip("set KOKORO_NATIVE_TRT_BUILD_TESTS=1 to run native compile tests")

    output_dir = tmp_path / f"artifact-{precision}"
    compile_artifact(
        output_dir,
        repo_id=os.getenv("KOKORO_TRT_REPO_ID", "hexgrad/Kokoro-82M"),
        model=os.getenv("KOKORO_TRT_MODEL"),
        precision=precision,
        include_voices=[],
    )

    artifact = TensorRTArtifact.load(output_dir)
    assert artifact.metadata.precision == precision
    assert artifact.paths.engine_path.is_file()
    assert artifact.paths.onnx_path.is_file()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="native TensorRT tests require CUDA"
)
def test_build_engine_parses_onnx_with_external_weights(tmp_path):
    if os.getenv("KOKORO_NATIVE_TRT_BUILD_TESTS") != "1":
        pytest.skip("set KOKORO_NATIVE_TRT_BUILD_TESTS=1 to run native compile tests")

    numpy = pytest.importorskip("numpy")
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("tensorrt")

    from onnx import helper, numpy_helper

    onnx_path = tmp_path / "external-data.onnx"
    weights_path = tmp_path / "external-data.weights"
    engine_path = tmp_path / "external-data.plan"

    x = helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 1])
    audio = helper.make_tensor_value_info("audio", onnx.TensorProto.FLOAT, [1, 1])
    weight = numpy_helper.from_array(
        numpy.asarray([[2.0]], dtype=numpy.float32),
        name="weight",
    )
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "weight"], ["audio"])],
        "external-data",
        [x],
        [audio],
        initializer=[weight],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 18)],
    )
    onnx.save_model(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=weights_path.name,
        size_threshold=0,
        convert_attribute=False,
    )

    assert onnx_path.is_file()
    assert weights_path.is_file()

    shapes = {
        "lower": {"x": (1, 1)},
        "preferred": {"x": (1, 1)},
        "upper": {"x": (1, 1)},
    }
    build_engine_from_onnx(
        onnx_path=onnx_path,
        engine_path=engine_path,
        shapes=shapes,
        input_order=("x",),
        precision="fp32",
        workspace_size=None,
        builder_optimization_level=None,
    )

    assert engine_path.is_file()
    assert engine_path.stat().st_size > 0


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="native TensorRT tests require CUDA"
)
def test_native_engine_runs_random_profile_points():
    artifact = TensorRTArtifact.load(existing_artifact_dir())
    engine = NativeTRTEngine(artifact.paths.engine_path)
    sink = InMemoryTraceSink()
    telemetry = Telemetry(ProfilerConfig(enabled=True), [sink])

    for group in ("lower", "preferred", "upper"):
        chunk = telemetry.start_chunk()
        inputs = {
            name: torch.randn(
                tuple(shape),
                device="cuda",
                dtype=(
                    torch.float16
                    if artifact.metadata.precision == "fp16"
                    else torch.float32
                ),
            ).contiguous()
            for name, shape in artifact.metadata.shapes[group].items()
        }

        outputs = engine.run(inputs, profile=chunk)
        chunk.finalize("ok")
        assert set(outputs) == {"audio"}
        assert outputs["audio"].is_cuda
        assert outputs["audio"].numel() > 0

    stage_names = {stage.name for trace in sink.traces for stage in trace.stages}
    assert "trt.validate_inputs" in stage_names
    assert "trt.set_input_shapes" in stage_names
    assert "trt.infer_shapes" in stage_names
    assert "trt.allocate_outputs" in stage_names
    assert "trt.set_tensor_addresses" in stage_names
    assert "trt.execute_async_v3" in stage_names


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="native TensorRT tests require CUDA"
)
def test_native_engine_matches_pytorch_generator_at_preferred_shape():
    artifact = TensorRTArtifact.load(existing_artifact_dir())
    model = load_model_for_artifact(artifact).to("cuda").eval()
    plan = ShapePlan.from_model(model)
    shapes = plan.engine_shapes(model)["preferred"]

    dtype = torch.float16 if artifact.metadata.precision == "fp16" else torch.float32
    torch_generator = (
        GeneratorExportBuilder.build(model).to(device="cuda", dtype=dtype).eval()
    )
    native_engine = NativeTRTEngine(artifact.paths.engine_path)

    inputs = {
        name: torch.randn(tuple(shapes[name]), device="cuda", dtype=dtype).contiguous()
        for name in plan.input_order()
    }

    with torch.inference_mode():
        expected = torch_generator(
            inputs["x"],
            inputs["ref_s"],
            *(inputs[f"source_{i}"] for i in range(len(plan.source_channels))),
        ).float()

    actual = native_engine.run(inputs)["audio"].float()
    torch.cuda.current_stream().synchronize()

    if artifact.metadata.precision == "fp16":
        assert torch.allclose(actual, expected, rtol=5e-2, atol=5e-2)
    else:
        assert torch.allclose(actual, expected, rtol=1e-3, atol=1e-3)
